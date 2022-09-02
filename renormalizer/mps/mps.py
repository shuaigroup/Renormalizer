# -*- encoding: utf-8 -*-

import logging
from collections import Counter, deque
from functools import wraps, reduce
from typing import Union, List, Dict
import itertools


import scipy
from scipy import stats

from renormalizer.lib import solve_ivp, expm_krylov
from renormalizer.model import Model, Op, basis as ba
from renormalizer.mps import svd_qn
from renormalizer.mps.backend import backend, np, xp
from renormalizer.mps.lib import (
    Environ,
    select_basis,
    compressed_sum,
    contract_one_site,
    cvec2cmat
)
from renormalizer.mps.matrix import (
    multi_tensor_contract,
    ones,
    tensordot,
    Matrix,
    asnumpy,
    asxp)
from renormalizer.mps.mp import MatrixProduct
from renormalizer.mps.hop_expr import hop_expr
from renormalizer.mps.mpo import Mpo
from renormalizer.utils import (
    OptimizeConfig,
    CompressCriteria,
    EvolveConfig,
    EvolveMethod
)
from renormalizer.utils.utils import calc_vn_entropy

logger = logging.getLogger(__name__)


def adaptive_tdvp(fun):
    # evolve t/2 (twice) and t to obtain the O(dt^3) error term in 2nd-order Trotter decomposition
    # J. Chem. Phys. 146, 174107 (2017)

    @wraps(fun)
    def adaptive_fun(self: "Mps", mpo, evolve_target_t):

        if not self.evolve_config.adaptive:
            return fun(self, mpo, evolve_target_t)
        config: EvolveConfig = self.evolve_config.copy()
        config.check_valid_dt(evolve_target_t)

        cur_mps = self
        # prevent bug
        del self

        # setup some constant
        p_restart = 0.5  # restart threshold
        p_min = 0.1  # safeguard for minimal allowed p
        p_max = 2.  # safeguard for maximal allowed p

        evolved_t = 0

        while True:

            dt = min_abs(config.guess_dt, evolve_target_t - evolved_t)
            logger.debug(
                    f"guess_dt: {config.guess_dt}, try time step size: {dt}"
            )
            
            mps_half1 = fun(cur_mps, mpo, dt / 2)
            mps_half2 = fun(mps_half1, mpo, dt / 2)
            mps = fun(cur_mps, mpo, dt)
            dis = mps.distance(mps_half2)

            # prevent bug. save "some" memory.
            del mps_half1, mps

            p = (0.75 * config.adaptive_rtol / (dis/mps_half2.mp_norm + 1e-30)) ** (1./3)
            logger.debug(f"distance: {dis}, enlarge p parameter: {p}")
            if p < p_min:
                p = p_min
            if p_max < p:
                p = p_max

            # rejected
            if p < p_restart:
                config.guess_dt = dt * p
                logger.debug(
                    f"evolution not converged, new guess_dt: {config.guess_dt}"
                )
                continue

            # accepted
            evolved_t += dt
            if np.allclose(evolved_t, evolve_target_t):
                # normal exit. Note that `dt` could be much less than actually tolerated for the last step
                # so use `guess_dt` for the last step. Slight inaccuracy won't harm.
                mps_half2.evolve_config.guess_dt = config.guess_dt
                logger.debug(
                    f"evolution converged, new guess_dt: {mps_half2.evolve_config.guess_dt}"
                )
                return mps_half2
            else:
                # in this case `config.guess_dt == dt`
                config.guess_dt *= p
                logger.debug(f"sub-step {dt} further, evolved: {evolved_t}, new guess_dt: {config.guess_dt}")
                cur_mps = mps_half2

    return adaptive_fun


class Mps(MatrixProduct):
    @classmethod
    def random(cls, model: Model, nexciton, m_max, percent=1.0) -> "Mps":
        # a high percent makes the result more random
        # sometimes critical for getting correct optimization result
        mps = cls()
        mps.model = model
        mps.qn = [[0]]
        dim_list = [1]

        for imps in range(model.nsite - 1):

            # quantum number
            qnbig = np.add.outer(mps.qn[imps], mps._get_sigmaqn(imps)).flatten()
            u_set = []
            s_set = []
            qnset = []
            
            # this random state is only suitable for positive quantum number is 0,1,2,3...
            for iblock in range(min(qnbig), nexciton + 1):
                # find the quantum number index
                indices = [i for i, x in enumerate(qnbig) if x == iblock]

                if len(indices) != 0:
                    a: np.ndarray = np.random.random([len(indices), len(indices)]) - 0.5
                    a = a + a.T
                    s, u = scipy.linalg.eigh(a=a)
                    u_set.append(svd_qn.blockrecover(indices, u, len(qnbig)))
                    s_set.append(s)
                    qnset += [iblock] * len(indices)

            u_set = np.concatenate(u_set, axis=1)
            s_set = np.concatenate(s_set)
            mt, mpsdim, mpsqn, nouse = select_basis(
                u_set, s_set, qnset, u_set, m_max, percent=percent
            )
            # add the next mpsdim
            dim_list.append(mpsdim)
            mps.append(mt.reshape((dim_list[imps], -1, dim_list[imps + 1])))
            mps.qn.append(mpsqn)

        # the last site
        mps.qn.append([0])
        dim_list.append(1)
        last_mt = (
            xp.random.random([dim_list[-2], mps.pbond_list[-1], dim_list[-1]]) - 0.5
        )
        # normalize the mt so that the whole mps is normalized
        last_mt /= xp.linalg.norm(last_mt.flatten())
        mps.append(last_mt)

        mps.qnidx = len(mps) - 1
        mps.to_right = False
        mps.qntot = nexciton

        return mps

    @classmethod
    def hartree_product_state(cls, model, condition: Dict):
        r"""
        Construct a Hartree product state
        
        Args:
            model (:class:`~renormalizer.model.Model`): Model information.
            condition (Dict): Dict with format ``{dof:local_state}``.
                The default local state for dofs not specified is the "0" state.
                An example is ``{"e_1":1, "v_0":2, "v_3":[0, 0.707, 0.707]}``.

                Note:
                    If there are bases that contain multiple dofs in the model, the value of the dict
                    is the state of all dofs of the basis. For example,
                    if a basis contains ``"e_1"``, ``"e_2"`` and ``"e_3"``,
                    ``{"e_1": 2}`` (``{"e_1": [0, 0, 1]}``) means ``"e_3"`` is occupied and
                    ``{"e_1": 1}`` (``{"e_1": [0, 1, 0]}``) means ``"e_2"`` is occupied.
                    Be aware that in :class:`renormalizer.utils.basis.BasisMultiElectronVac` the vacuum state
                    is added to the ``0`` index.

        Returns:
            Constructed mps (:class:`Mps`)
        """
        
        mps = cls()
        mps.model = model
        mps.build_empty_mp(model.nsite)
        qn_single = [[],] * model.nsite

        # check that the condition is not duplicated
        # each site has at most 1 single key to assign the occupation  
        index = [model.dof_to_siteidx[key] for key in condition.keys()]
        assert len(index) == len(set(index))
        # replace the dof_name key to site_index key
        condition = {model.dof_to_siteidx[key]:value for key, value in
                condition.items()}

        for isite, local_basis in enumerate(model.basis):
            pdim = local_basis.nbas
            ms = np.zeros((1, pdim, 1))
            local_state = condition.pop(isite,0)
            if isinstance(local_state, int):
                ms[0, local_state, 0] = 1.
                qn = local_basis.sigmaqn[local_state]
            else:
                ms[0, :, 0] = local_state
                # quantum numbers for all states occupied
                all_qn = np.array(local_basis.sigmaqn)[np.nonzero(local_state)]
                if all_qn.std() != 0:
                    raise ValueError("Quantum numbers are mixed in the condition.")
                qn = all_qn[0]

            mps[isite] = ms
            qn_single[isite] = qn

        if len(condition) != 0:
            raise ValueError(f"Condition not complete used: {condition}")
        qn_single = np.array(qn_single)
        mps.qn = [[0]]
        for isite in range(model.nsite):
            mps.qn.append([np.sum(qn_single[:isite+1])])
        mps.qnidx = model.nsite - 1
        mps.qntot = mps.qn[-1][0]
        mps.qn[-1] = [0]
        mps.to_right = False

        return mps

    @classmethod
    def ground_state(cls, model: Model, max_entangled: bool,
            normalize:bool=True, condition:Dict=None):
        r"""
        Obtain ground state at :math:`T = 0` or :math:`T = \infty` (maximum entangled).
        Electronic DOFs are always at ground state. and vibrational DOFs depend on ``max_entangled``.
        For Spin-Boson model the electronic DOF also depends on ``max_entangled``.
        

        Parameters
        ----------
            model : :class:`~renormalizer.model.Model`
                system information.
            max_entangled : bool
                temperature of the vibrational DOFs. If set to ``True``,
                :math:`T = \infty` and if set to ``False``, :math:`T = 0`.
            normalize: bool, optional
                if the returned Mps are normalized when ``max_entangled=True``.
                Default is True. If ``normalize=False``, the vibrational part is identity.
            condition: dict, optional
                the same as `hartree_product_state`. only used in ba.BasisMultiElectron
                cases to define the occupation. Default is ``None``.
        Returns
        -------
            mps : renormalizer.mps.Mps
        
        """
        
        mps = cls()
        mps.model = model
        mps.qn = [[0]] * (model.nsite + 1)
        mps.qnidx = model.nsite - 1
        mps.to_right = False
        mps.qntot = 0
            
        mps.build_empty_mp(model.nsite)
        
        if condition is not None:
            # check that the condition is not duplicated
            # each site has at most 1 single key to assign the occupation  
            index = [model.dof_to_siteidx[key] for key in condition.keys()]
            assert len(index) == len(set(index))
            # replace the dof_name key to site_index key
            condition = {model.dof_to_siteidx[key]:value for key, value in
                    condition.items()}

        for isite, local_basis in enumerate(model.basis):
            pdim = local_basis.nbas
            ms = np.zeros((1, pdim, 1))
            if local_basis.is_phonon:
                if max_entangled:
                    if normalize:
                        ms[0, :, 0] = 1.0 / np.sqrt(pdim)
                    else:
                        ms[0, :, 0] = 1.0
                else:
                    ms[0, 0, 0] = 1.0
                mps[isite] = ms

            elif local_basis.is_electron or local_basis.is_spin:

                if isinstance(local_basis, ba.BasisSimpleElectron):
                    # simple electron site
                    ms[0,0,0] = 1.
                elif isinstance(local_basis, ba.BasisMultiElectron):
                    assert condition is not None
                    local_state = condition.pop(isite)
                    if isinstance(local_state, int):
                        ms[0, local_state, 0] = 1.
                        qn = local_basis.sigmaqn[local_state]
                    else:
                        ms[0, :, 0] = local_state
                        qn = local_basis.sigmaqn[np.nonzero(local_state)]
                    assert np.allclose(qn, 0)
                    if max_entangled and normalize:
                        ms /= np.linalg.norm(ms)

                elif isinstance(local_basis, ba.BasisMultiElectronVac):
                        ms[0,0,0] = 1.
                elif isinstance(local_basis, ba.BasisHalfSpin):
                    if max_entangled:
                        if normalize:
                            ms[0,:,0] = 1. / np.sqrt(2.)
                        else:
                            ms[0,:,0] = 1.
                    else:
                        ms[0,0,0] = 1.
                else:
                    raise NotImplementedError
                mps[isite] = ms
        for ms in mps:
            assert ms is not None
        return mps

    @classmethod
    def load(cls, model: Model, fname: str):
        npload = np.load(fname, allow_pickle=True)
        mp = cls()
        mp.model = model
        for i in range(int(npload["nsites"])):
            mt = npload[f"mt_{i}"]
            if np.iscomplexobj(mt):
                mp.dtype = backend.complex_dtype
            else:
                mp.dtype = backend.real_dtype
            mp.append(mt)
        mp.qn = npload["qn"]
        mp.qnidx = int(npload["qnidx"])
        mp.qntot = int(npload["qntot"])
        version = npload["version"]
        if version == "0.1":
            mp.to_right = bool(npload["left"])
            # in this protocol, TDH and coeff is not dumped
            logger.warning("Using old dump/load protocol. TD Hartree part will be lost")
            mp.coeff = 1
        elif version == "0.2":
            mp.to_right = bool(npload["to_right"])
            # in this protocol, TDH is dumped, but it's not useful anymore
            logger.warning("Using old dump/load protocol. TD Hartree part will be lost")
            mp.coeff = npload["tdh_wfns"][-1]
        elif version == "0.3":
            mp.to_right = bool(npload["to_right"])
            mp.coeff = npload["coeff"].item(0)
        else:
            raise ValueError(f"Unknown dump version: {version}")
        return mp

    @classmethod
    def from_dense(cls, model, wfn: np.ndarray):
        # for debugging
        mp = cls()
        mp.model = model
        if np.iscomplexobj(wfn):
            mp.dtype = backend.complex_dtype
        else:
            mp.dtype = backend.real_dtype
        residual_wfn = wfn.reshape([1] + [b.nbas for b in model.basis] + [1])
        for i in range(len(model.basis) - 1):
            wfn_2d = residual_wfn.reshape(residual_wfn.shape[0] * residual_wfn.shape[1], -1)
            q, r = np.linalg.qr(wfn_2d)
            mp.append(q.reshape(residual_wfn.shape[0], residual_wfn.shape[1], q.shape[1]))
            residual_wfn = r.reshape([r.shape[0]] + list(residual_wfn.shape[2:]))
        assert residual_wfn.ndim == 3
        mp.append(residual_wfn)
        mp.build_empty_qn()
        return mp

    def __init__(self):
        super().__init__()
        self.coeff: Union[float, complex] = 1

        self.optimize_config: OptimizeConfig = OptimizeConfig()
        self.evolve_config: EvolveConfig = EvolveConfig()

    def conj(self) -> "Mps":
        new_mps = super().conj()
        new_mps.coeff = new_mps.coeff.conjugate()
        return new_mps

    def to_complex(self, inplace=False) -> "Mps":
        new_mp = super(Mps, self).to_complex(inplace=inplace)
        new_mp.coeff = complex(new_mp.coeff)
        return new_mp

    def _get_sigmaqn(self, idx):
        return self.model.basis[idx].sigmaqn

    @property
    def is_mps(self):
        return True

    @property
    def is_mpo(self):
        return False

    @property
    def is_mpdm(self):
        return False

    @property
    def nexciton(self):
        return self.qntot

    @property
    def norm(self):
        '''the norm of the total wavefunction
        '''
        return np.linalg.norm(self.coeff) * self.mp_norm

    def _expectation_path(self):
        # S--a--S--e--S
        # |     |     |
        # |     d     |
        # |     |     |
        # O--b--O--g--O
        # |     |     |
        # |     f     |
        # |     |     |
        # S--c--S--h--S
        path = [
            ([0, 1], "abc, cfh -> abfh"),
            ([3, 0], "abfh, bdfg -> ahdg"),
            ([2, 0], "ahdg, ade -> hge"),
            ([1, 0], "hge, egh -> "),
        ]
        return path

    def _expectation_conj(self):
        return self.conj()

    def expectation(self, mpo, self_conj=None) -> Union[float, complex]:
        if self_conj is None:
            self_conj = self._expectation_conj()
        environ = Environ(self, mpo, "R", mps_conj=self_conj)
        l = xp.ones((1, 1, 1), dtype=self.dtype)
        r = environ.read("R", 1)
        path = self._expectation_path()
        val = multi_tensor_contract(path, l, self[0], mpo[0], self_conj[0], r)
        if np.isclose(float(val.imag), 0):
            return float(val.real)
        else:
            return complex(val)
        # This is time and memory consuming
        # return self_conj.dot(mpo.apply(self)).real

    def expectations(self, mpos, self_conj=None, opt=True) -> np.ndarray:

        if not opt:
            # the naive way, slow and time consuming. Yet predictable and reliable
            return np.array([self.expectation(mpo, self_conj) for mpo in mpos])

        # optimized way, cache for intermediates
        # hash is used as indices of the matrices.
        # The chance for collision (the same hash for two different matrices) is
        # about 1-0.99999999999997 in 1000 matrices.
        # In which case a RuntimeError is raised and rerun the job should solve the problem
        hash_to_obj = dict()
        mpos_hash: List[List] = []
        for mpo in mpos:
            mpo_hash = []
            for m in mpo:
                m_hash = hash(m)
                if m_hash not in hash_to_obj:
                    hash_to_obj[m_hash] = m
                else:
                    if not np.allclose(hash_to_obj[m_hash], m.array):
                        raise RuntimeError("Rare hash collision")
                mpo_hash.append(m_hash)
            mpos_hash.append(mpo_hash)

        if self_conj is None:
            self_conj = self._expectation_conj()
        l_environ_dict = _construct_freq_environ(mpos_hash, hash_to_obj, self, "L", self_conj)
        r_environ_dict = _construct_freq_environ(mpos_hash, hash_to_obj, self, "R", self_conj)
        results = []
        for mpo in mpos:
            l_environ, l_idx = _get_freq_environ(l_environ_dict, mpo, "L", np.inf)
            r_environ, r_idx = _get_freq_environ(r_environ_dict, mpo, "R", len(mpo)-l_idx-1)
            for i in range(l_idx+1, r_idx):
                l_environ = contract_one_site(l_environ, self[i], mpo[i], "L", self_conj[i])
            results.append(complex(l_environ.flatten() @ r_environ.flatten()))  # cast to python type

        results = np.array(results)
        if np.allclose(results.imag, 0):
            return results.real
        else:
            return results

    @property
    def ph_occupations(self):
        r"""
        phonon occupations :math:`b^\dagger_i b_i` for each electronic DoF.
        The order is defined by :attr:`~renormalizer.model.model.v_dofs`.
        """
        key = "ph_occupations"
        # ph_occupations is actually the occupation of the basis
        if key not in self.model.mpos:
            mpos = []
            for dof in self.model.v_dofs:
                mpos.append(Mpo(self.model, Op("n", dof)))
            self.model.mpos[key] = mpos
        else:
            mpos = self.model.mpos[key]

        return self.expectations(mpos)

    @property
    def e_occupations(self):
        r"""
        Electronic occupations :math:`a^\dagger_i a_i` for each electronic DoF.
        The order is defined by :attr:`~renormalizer.model.model.e_dofs`.
        """
        key = "e_occupations"
        if key not in self.model.mpos:
            mpos = []
            for dof in self.model.e_dofs:
                mpos.append(Mpo(self.model, Op(r"a^\dagger a", dof)))
            self.model.mpos[key] = mpos
        else:
            mpos = self.model.mpos[key]
        return self.expectations(mpos)

    def metacopy(self) -> "Mps":
        new: Mps = super().metacopy()
        new.coeff = self.coeff
        new.optimize_config = self.optimize_config
        # evolve_config has its own data
        new.evolve_config = self.evolve_config.copy()
        return new

    def normalize(self, kind):
        r''' normalize the wavefunction

        Parameters
        ----------
        kind: str
            "mps_only": the mps part is normalized and coeff is not modified;
            "mps_norm_to_coeff": the mps part is normalized and the norm is multiplied to coeff;
            "mps_and_coeff": both mps and coeff is normalized

        Returns
        -------
        ``self`` is overwritten.
        '''

        if kind == "mps_only":
            new_coeff = self.coeff
        elif kind == "mps_and_coeff":
            new_coeff = self.coeff / np.linalg.norm(self.coeff)
        elif kind == "mps_norm_to_coeff":
            new_coeff = self.coeff * self.mp_norm
        else:
            raise ValueError(f"kind={kind} is not valid.")
        new_mps = self.scale(1.0 / self.mp_norm, inplace=True)
        new_mps.coeff = new_coeff

        return new_mps


    def expand_bond_dimension(self, hint_mpo=None, coef=1e-10, include_ex=True):
        """
        expand bond dimension as required in compress_config
        """
        # expander m target
        m_target = self.compress_config.bond_dim_max_value - self.bond_dims_mean
        # will be restored at exit
        self.compress_config.bond_dim_max_value = m_target
        if self.compress_config.criteria is not CompressCriteria.fixed:
            logger.warning("Setting compress criteria to fixed")
            self.compress_config.criteria = CompressCriteria.fixed
        logger.debug(f"target for expander: {m_target}")
        if hint_mpo is None:
            expander = self.__class__.random(self.model, 1, m_target)
        else:
            # fill states related to `hint_mpo`
            logger.debug(
                f"average bond dimension of hint mpo: {hint_mpo.bond_dims_mean}"
            )
            # in case of localized `self`
            if include_ex:
                if self.is_mps:
                    ex_state: MatrixProduct = self.ground_state(self.model, False)
                    # for self.qntot >= 1
                    for i in range(self.qntot):
                        ex_state = Mpo.onsite(self.model, r"a^\dagger") @ ex_state
                elif self.is_mpdm:
                    assert self.qntot == 1
                    ex_state: MatrixProduct = self.max_entangled_ex(self.model)
                else:
                    assert False
                ex_state.compress_config = self.compress_config
                ex_state.move_qnidx(self.qnidx)
                ex_state.to_right = self.to_right
                lastone = self + ex_state

            else:
                lastone = self
            expander_list: List["MatrixProduct"] = []
            cumulated_m = 0
            while True:
                lastone.compress_config.criteria = CompressCriteria.fixed
                expander_list.append(lastone)
                expander = compressed_sum(expander_list)
                if cumulated_m == expander.bond_dims_mean:
                    # probably a small system, the required bond dimension can't be reached
                    break
                cumulated_m = expander.bond_dims_mean
                logger.debug(
                    f"cumulated bond dimension: {cumulated_m}. lastone bond dimension: {lastone.bond_dims}"
                )
                if m_target < cumulated_m:
                    break
                if m_target < 0.8 * (lastone.bond_dims_mean * hint_mpo.bond_dims_mean):
                    lastone = lastone.canonicalise().compress(
                        m_target // hint_mpo.bond_dims_mean + 1
                    )
                lastone = (hint_mpo @ lastone).normalize("mps_and_coeff")
        logger.debug(f"expander bond dimension: {expander.bond_dims}")
        self.compress_config.bond_dim_max_value += self.bond_dims_mean
        return (self + expander.scale(coef*self.norm, inplace=True)).canonicalise().canonicalise().normalize("mps_norm_to_coeff")

    def evolve(self, mpo, evolve_dt, normalize=True) -> "Mps":

        method = {
            EvolveMethod.prop_and_compress: self._evolve_prop_and_compress,
            EvolveMethod.prop_and_compress_tdrk4: self._evolve_prop_and_compress_tdrk4,
            EvolveMethod.prop_and_compress_tdrk: self._evolve_prop_and_compress_tdrk,
            EvolveMethod.tdvp_mu_vmf: self._evolve_tdvp_mu_vmf,
            EvolveMethod.tdvp_vmf: self._evolve_tdvp_mu_vmf,
            EvolveMethod.tdvp_mu_cmf: self._evolve_tdvp_mu_cmf,
            EvolveMethod.tdvp_ps: self._evolve_tdvp_ps,
            EvolveMethod.tdvp_ps2: self._evolve_tdvp_ps2
        }[self.evolve_config.method]
        new_mps = method(mpo, evolve_dt)
        if normalize:
            if np.iscomplex(evolve_dt):
                new_mps.normalize("mps_and_coeff")
            else:
                new_mps.normalize("mps_only")
        return new_mps
    
    def _evolve_prop_and_compress_tdrk4(self, mpo, evolve_dt) -> "Mps":
        """
        classical 4th order Runge-Kutta solver for time-dependent Hamiltonian
        """
        
        if isinstance(mpo, Mpo): 
            def mpo_t(t, *args, **kwargs):
                return mpo
        elif callable(mpo):
            # mpo can be a function of time, the range is 0 -> evolve_dt
            mpo_t = mpo
        else:
            raise TypeError(f"unsupported mpo type: {mpo}")

        k1 = mpo_t(0).contract(self).scale(-1j)
        logger.debug(f"k1:{k1}")  
        tmp_mps = self + k1.scale(0.5*evolve_dt)
        tmp_mps.canonicalise().compress()
        k2 = mpo_t(0.5*evolve_dt).contract(tmp_mps).scale(-1j)
        logger.debug(f"k2:{k2}")  
        tmp_mps = self + k2.scale(0.5*evolve_dt)
        tmp_mps.canonicalise().compress()
        k3 = mpo_t(0.5*evolve_dt).contract(tmp_mps).scale(-1j)
        logger.debug(f"k3:{k3}")  
        tmp_mps = self + k3.scale(evolve_dt)
        tmp_mps.canonicalise().compress()
        k4 = mpo_t(evolve_dt).contract(tmp_mps).scale(-1j)
        logger.debug(f"k4:{k4}")  
        
        new_mps = compressed_sum([self, k1.scale(1/6*evolve_dt),
            k2.scale(2/6*evolve_dt), 
            k3.scale(2/6*evolve_dt),
            k4.scale(1/6*evolve_dt)])
        logger.info(f"new_mps:{new_mps}")  
        
        return new_mps
    
    def _evolve_prop_and_compress_tdrk(self, mpo, evolve_dt) -> "Mps":
        """
            The most general Runge-Kutta solver for both time-dependent and
            time-independnet Hamiltonian and adaptive or unadaptive time-step
            size evolution
        """
        if isinstance(mpo, Mpo): 
            def mpo_t(t, *args, **kwargs):
                return mpo
        elif callable(mpo):
            mpo_t = mpo
        else:
            raise TypeError(f"unsupported mpo type: {mpo}")
        
        rk_config = self.evolve_config.rk_config
        a,b,c = rk_config.tableau
        
        def sub_time_step_evolve(y,tau,t0):
            # error is relative error
            k_list = []
            for istage in range(rk_config.stage):
                k = compressed_sum([y]+[k_list[i].scale(a[istage,i]*tau) for
                    i in range(istage) if a[istage,i] != 0], batchsize=6)
                k = mpo_t(c[istage]*tau+t0, mps=k).contract(k).scale(-1j)
                logger.debug(f"k_{istage}: {k}") 
                k_list.append(k)        
            
            new_mps = compressed_sum([y] +
                    [k_list[istage].scale(b[0,istage]*tau) \
                    for istage in range(rk_config.stage) if b[0,istage]!=0],
                    batchsize=6)   
            logger.debug(f"order_{rk_config.order[0]}: {new_mps}") 
            
            if self.evolve_config.adaptive:
                assert len(rk_config.order) == 2
                assert rk_config.order[0] - rk_config.order[1] == 1
                error = reduce(lambda mps1, mps2: mps1.add(mps2),
                    [k_list[istage].scale((b[0,istage]-b[1,istage])*tau) \
                    for istage in range(rk_config.stage) if not \
                    np.allclose(b[0,istage],b[1,istage])])

                error = error.norm / new_mps.norm
            else:
                assert len(rk_config.order) == 1
                error = 0
                
            return new_mps, error
        
        self.evolve_config.check_valid_dt(evolve_dt)
        
        if self.evolve_config.adaptive:
            p_restart = 0.5  # restart threshold
            p_min = 0.1  # safeguard for minimal allowed p
            p_max = 2.0  # safeguard for maximal allowed p
            
            evolved_dt = 0
            new_mps = self

            while True:
                dt = min_abs(new_mps.evolve_config.guess_dt, evolve_dt-evolved_dt)
                logger.debug(f"guess_dt: {new_mps.evolve_config.guess_dt}, try time step size: {dt}")
                new_mps, error = sub_time_step_evolve(new_mps, dt, evolved_dt)    
                p = (new_mps.evolve_config.adaptive_rtol / (error + 1e-30)) ** (1/rk_config.order[0])
                logger.debug(f"RKsolver:{rk_config.method} relative error: {error}, enlarge p parameter: {p}")
                
                if p < p_restart:
                    # not accurate, will restart
                    new_mps.evolve_config.guess_dt = dt * max(p_min, p)
                    logger.debug(
                        f"evolution not converged, new guess_dt: {new_mps.evolve_config.guess_dt}"
                    )
                else:
                    if xp.allclose(dt+evolved_dt, evolve_dt):
                        new_mps.evolve_config.guess_dt = min_abs(
                            dt * p, new_mps.evolve_config.guess_dt
                        )
                        # normal exit
                        logger.debug(
                            f"evolution converged, new guess_dt: {new_mps.evolve_config.guess_dt}"
                        )
                        break
                    else:
                        new_mps.evolve_config.guess_dt *= min(p, p_max)
                        evolved_dt += dt
                        logger.debug(
                            f"evolution converged, new guess_dt: {new_mps.evolve_config.guess_dt}"
                        )
                        logger.debug(f"sub-step {dt} further, remaining: {evolve_dt-evolved_dt}")
        else:
            new_mps, _ = sub_time_step_evolve(self, evolve_dt, 0)

        return new_mps

    def _evolve_prop_and_compress(self, mpo, evolve_dt) -> "Mps":
        """
        The global propagation & compression evolution scheme
        only for time-independent Hamiltonian
        Taylor expansion approximation to the formal propagator
        """
        config = self.evolve_config
        assert evolve_dt is not None

        propagation_c = config.taylor_config.coeff
        order = len(propagation_c)-1
        termlist = [self]
        # don't let bond dim grow when contracting
        orig_compress_config = self.compress_config
        contract_compress_config = self.compress_config.copy()
        if contract_compress_config.criteria is CompressCriteria.threshold:
            contract_compress_config.criteria = CompressCriteria.both
        # contract_compress_config.min_dims = None
        # contract_compress_config.max_dims = np.array(self.bond_dims) + 4
        self.compress_config = contract_compress_config

        while len(termlist) < len(propagation_c):
            termlist.append(mpo.contract(termlist[-1]))
        # bond dim can grow after adding
        for t in termlist:
            t.compress_config = orig_compress_config

        if config.adaptive:
            config.check_valid_dt(evolve_dt)

            p_restart = 0.5  # restart threshold
            p_min = 0.1  # safeguard for minimal allowed p
            p_max = 2.0  # safeguard for maximal allowed p

            while True:
                scaled_termlist = []
                dt = min_abs(config.guess_dt, evolve_dt)
                logger.debug(f"guess_dt: {config.guess_dt}, try time step size: {dt}")
                for idx, term in enumerate(termlist):
                    scale = (-1.0j * dt) ** idx * propagation_c[idx]
                    scaled_termlist.append(term.scale(scale))
                    del term
                
                new_mps1 = compressed_sum(scaled_termlist[:-1])
                new_mps2 = compressed_sum(
                    [new_mps1, scaled_termlist[-1]]
                )
                dis = new_mps1.distance(new_mps2)
                p = (config.adaptive_rtol / (dis/new_mps2.mp_norm + 1e-30)) ** (1/order)
                logger.debug(f"RK45 error distance: {dis}, enlarge p parameter: {p}")

                if xp.allclose(dt, evolve_dt):
                    # approahes the end
                    if p < p_restart:
                        # not accurate in this final sub-step will restart
                        config.guess_dt = dt * max(p_min, p)
                        logger.debug(
                            f"evolution not converged, new guess_dt: {config.guess_dt}"
                        )
                    else:
                        # normal exit
                        new_mps2.evolve_config.guess_dt = min_abs(
                            dt * p, config.guess_dt
                        )
                        logger.debug(
                            f"evolution converged, new guess_dt: {new_mps2.evolve_config.guess_dt}"
                        )
                        return new_mps2
                else:
                    # sub-steps
                    if p < p_restart:
                        config.guess_dt *= max(p_min, p)
                        logger.debug(
                            f"evolution not converged, new guess_dt: {config.guess_dt}"
                        )
                    else:
                        new_dt = evolve_dt - dt
                        config.guess_dt *= min(p, p_max)
                        new_mps2.evolve_config.guess_dt = config.guess_dt
                        # memory consuming and not useful anymore
                        del new_mps1, termlist, scaled_termlist
                        logger.debug(
                            f"evolution converged, new guess_dt: {config.guess_dt}"
                        )
                        logger.debug(f"sub-step {dt} further, remaining: {new_dt}")
                        return new_mps2._evolve_prop_and_compress(mpo, new_dt)
        else:
            for idx, term in enumerate(termlist):
                term.scale(
                    (-1.0j * evolve_dt) ** idx * propagation_c[idx], inplace=True
                )
            return compressed_sum(termlist)
    
    def _evolve_tdvp_mu_vmf(self, mpo, evolve_dt) -> "Mps":
        """
        variable mean field
        see the difference between VMF and CMF, refer to Z. Phys. D 42, 113–129 (1997)
        the matrix unfolding algorithm, see arXiv:1907.12044
        only the RKF45 integration is used.
        The default RKF45 local step error tolerance is rtol:1e-5, atol:1e-8
        regulation of S is 1e-10, these default parameters could be changed in
        /utils/configs.py

        """
        
        if isinstance(mpo, Mpo): 
            def mpo_t(t, *args, **kwargs):
                return mpo
        elif callable(mpo):
            mpo_t = mpo
        else:
            raise TypeError(f"unsupported mpo type: {mpo}")
        
        # a workaround for https://github.com/scipy/scipy/issues/10164
        imag_time = np.iscomplex(evolve_dt)
        if imag_time:
            evolve_dt = -evolve_dt.imag
            # used in calculating derivatives
            coef = -1
        else:
            coef = 1j
        
        # only not canonicalise when force_ovlp=True and to_right=False
        if not (self.evolve_config.force_ovlp and not self.to_right):
            self.ensure_left_canonical()

        # `self` should not be modified during the evolution
        if imag_time:
            mps = self.copy()
        else:
            mps = self.to_complex()

        # the quantum number symmetry is used
        qnmat_list = []
        position = [0]
        qntot = mps.qntot
        for imps in range(mps.site_num):
            mps.move_qnidx(imps)
            qnbigl, qnbigr, qnmat= mps._get_big_qn([imps])
            qnmat_list.append(qnmat)
            position.append(position[-1]+np.sum(qnmat == qntot))

        sw_min_list = []
        
        def func_vmf(t,y):
            
            sw_min_list.clear()

            # update mps: from left to right
            for imps in range(mps.site_num):
                mps[imps] = cvec2cmat(mps[imps].shape, asnumpy(y[position[imps]:position[imps + 1]]),
                                                           qnmat_list[imps], qntot)
            mpo = mpo_t(t, mps=mps)

            if self.evolve_config.method == EvolveMethod.tdvp_mu_vmf:
                environ_mps = mps.copy()
            elif self.evolve_config.method == EvolveMethod.tdvp_vmf:
                environ_mps = mps
                # the first S_R
                S_R = np.ones([1, 1], dtype=mps.dtype)
            else:
                assert False

            environ = Environ(environ_mps, mpo, "L")

            if self.evolve_config.force_ovlp:
                # construct the S_L list (type: Matrix) and S_L_inv list (type: xp.array)
                # len: mps.site_num+1
                S_L_list = [
                    np.ones([1, 1], dtype=mps.dtype),
                ]
                for imps in range(mps.site_num):
                    S_L_list.append(
                        transferMat(mps, None, "L", imps, S_L_list[imps])
                    )


                S_L_inv_list = []
                for imps in range(mps.site_num + 1):
                    w, u = scipy.linalg.eigh(S_L_list[imps])
                    S_L_inv = u.dot(np.diag(1.0 / w)).dot(u.T.conj())
                    S_L_inv_list.append(S_L_inv)
            else:
                S_L_list = [None,] * (mps.site_num + 1)
                S_L_inv_list = [None,] * (mps.site_num + 1)

            # calculate hop_y: from right to left
            hop_y = xp.empty_like(y)

            for imps in mps.iter_idx_list(full=True):
                shape = list(mps[imps].shape)
                ltensor = asxp(environ.read("L", imps - 1))

                if imps == self.site_num - 1:
                    # the coefficient site
                    rtensor = xp.ones((1, 1, 1), dtype=mps.dtype)
                    hop =hop_expr(ltensor, rtensor, [asxp(mpo[imps])], shape)

                    S_inv = xp.diag(xp.ones(1,dtype=mps.dtype))
                    func = integrand_func_factory(shape, hop, True, S_inv, True,
                            coef, ovlp_inv1=S_L_inv_list[imps+1],
                            ovlp_inv0=S_L_inv_list[imps], ovlp0=S_L_list[imps])

                    hop_y[position[imps]:position[imps+1]] = func(0,
                            mps[imps].array.ravel()).reshape(mps[imps].shape)[qnmat_list[imps]==qntot]

                    continue

                if self.evolve_config.method == EvolveMethod.tdvp_mu_vmf:
                    # perform qr on the environment mps
                    qnbigl, qnbigr, _ = environ_mps._get_big_qn([imps + 1])
                    u, s, qnlset, v, s, qnrset = svd_qn.svd_qn(
                            environ_mps[imps + 1].array, qnbigl, qnbigr,
                            environ_mps.qntot, system="R", full_matrices=False)
                    vt = v.T

                    environ_mps[imps + 1] = vt.reshape(environ_mps[imps + 1].shape)

                    rtensor = environ.GetLR(
                        "R", imps + 1, environ_mps, mpo, itensor=None, method="System"
                    )

                    sw_min_list.append(s.min())
                    regular_s = _mu_regularize(s, epsilon=self.evolve_config.reg_epsilon)

                    u = asxp(u)
                    us = u.dot(xp.diag(s))

                    rtensor = xp.tensordot(rtensor, us, axes=(-1, -1))

                    environ_mps[imps] = xp.tensordot(asxp(environ_mps[imps]), us, axes=(-1, 0))
                    environ_mps.qn[imps + 1] = qnrset
                    environ_mps.qnidx = imps

                    S_inv = u.conj().dot(xp.diag(1.0 / regular_s)).T

                elif self.evolve_config.method == EvolveMethod.tdvp_vmf:
                    rtensor = environ.GetLR(
                        "R", imps + 1, environ_mps, mpo, itensor=None, method="System")

                    # regularize density matrix
                    # Note that S_R is (#.conj, #)
                    S_R = transferMat(environ_mps, None, "R", imps + 1, S_R)
                    w, u = scipy.linalg.eigh(asnumpy(S_R))
                    # discard the negative eigenvalues due to numerical error
                    w = np.where(w>0, w, 0)

                    sw_min_list.append(w.min())

                    epsilon = self.evolve_config.reg_epsilon
                    w = w + epsilon * np.exp(-w / epsilon)

                    u = asxp(u)
                    # S_inv is (#.conj, #)
                    S_inv = u.dot(xp.diag(1.0 / w)).dot(u.T.conj()).T

                hop = hop_expr(ltensor, rtensor, [asxp(mpo[imps])], shape)

                func = integrand_func_factory(shape, hop, False, S_inv, True,
                        coef, ovlp_inv1=S_L_inv_list[imps+1],
                        ovlp_inv0=S_L_inv_list[imps], ovlp0=S_L_list[imps])

                hop_y[position[imps]:position[imps+1]] = func(0,
                        asxp(mps[imps].array.ravel())).reshape(mps[imps].shape)[qnmat_list[imps]==qntot]

            return hop_y

        init_y = xp.concatenate([asxp(ms.array[qnmat_list[ims]==qntot]) for ims, ms in enumerate(mps)])
        # the ivp local error, please refer to the Scipy default setting
        sol = solve_ivp(
            func_vmf,
            (0, evolve_dt),
            init_y,
            method="RK45",
            rtol=self.evolve_config.ivp_rtol,
            atol=self.evolve_config.ivp_atol,
        )

        # update mps: from left to right
        for imps in range(mps.site_num):
            mps[imps] = cvec2cmat(mps[imps].shape, asnumpy(sol.y[:, -1][position[imps]:position[imps + 1]]),
                                                       qnmat_list[imps], qntot)

        logger.info(f"{self.evolve_config.method} VMF func called: {sol.nfev}. RKF steps: {len(sol.t)}")

        sw_min_list = xp.array(sw_min_list)
        # auto-switch between tdvp_mu_vmf and tdvp_vmf
        if self.evolve_config.vmf_auto_switch:
            if sw_min_list.min() > np.sqrt(self.evolve_config.reg_epsilon*10.) and \
                mps.evolve_config.method == EvolveMethod.tdvp_mu_vmf:

                logger.debug(f"sw.min={sw_min_list.min()}, Switch to tdvp_vmf")
                mps.evolve_config.method =  EvolveMethod.tdvp_vmf

            elif sw_min_list.min() < self.evolve_config.reg_epsilon and \
                mps.evolve_config.method == EvolveMethod.tdvp_vmf:

                logger.debug(f"sw.min={sw_min_list.min()}, Switch to tdvp_mu_vmf")
                mps.evolve_config.method =  EvolveMethod.tdvp_mu_vmf

        # The caller do not want to deal with an MPS that is not canonicalised
        return mps.canonicalise()

    @adaptive_tdvp
    def _evolve_tdvp_mu_cmf(self, mpo, evolve_dt) -> "Mps":
        """
        evolution scheme: TDVP + constant mean field + matrix-unfolding
        regularization
        MPS :  LLLLLLC
        1st / 2nd order(default) CMF

        for 2nd order CMF:
        L is evolved with midpoint scheme
        C is evolved with midpoint(default) / trapz scheme
        """

        if self.evolve_config.tdvp_cmf_c_trapz:
            assert self.evolve_config.tdvp_cmf_midpoint

        imag_time = np.iscomplex(evolve_dt)

        # a workaround for https://github.com/scipy/scipy/issues/10164
        if imag_time:
            evolve_dt = -evolve_dt.imag
            # used in calculating derivatives
            coef = -1
        else:
            coef = 1j

        self.ensure_left_canonical()

        # `self` should not be modified during the evolution
        # mps: the mps to return
        # environ_mps: mps to construct environ
        if imag_time:
            mps = self.copy()
        else:
            mps = self.to_complex()

        if self.evolve_config.tdvp_cmf_midpoint:
            # mps at t/2 (1st order) as environment
            orig_config = self.evolve_config.copy()
            self.evolve_config.tdvp_cmf_midpoint = False
            self.evolve_config.tdvp_cmf_c_trapz = False
            self.evolve_config.adaptive = False
            environ_mps = self.evolve(mpo, evolve_dt / 2)
            self.evolve_config = orig_config
        else:
            # mps at t=0 as environment
            environ_mps = mps.copy()

        if self.evolve_config.tdvp_cmf_c_trapz:
            loop = 2
            mps[-1] = environ_mps[-1].copy()
        else:
            loop = 1

        while loop > 0:

            # construct the environment matrix
            environ = Environ(environ_mps, mpo, "L")

            # statistics for debug output
            cmf_rk_steps = []

            if self.evolve_config.force_ovlp:
                # construct the S_L list (type: Matrix) and S_L_inv list (type: xp.array)
                # len: mps.site_num+1
                S_L_list = [np.ones([1, 1], dtype=mps.dtype),]
                for imps in range(mps.site_num):
                    S_L_list.append(transferMat(environ_mps, None, "L", imps,
                        S_L_list[imps]))

                S_L_inv_list = []
                for imps in range(mps.site_num+1):
                    w, u = scipy.linalg.eigh(S_L_list[imps])
                    S_L_inv = xp.asarray(u.dot(np.diag(1.0 / w)).dot(u.T.conj()))
                    S_L_inv_list.append(S_L_inv)
                    S_L_list[imps] = S_L_list[imps]
            else:
                S_L_list = [None,] * (mps.site_num+1)
                S_L_inv_list = [None,] * (mps.site_num+1)

            for imps in mps.iter_idx_list(full=True):
                shape = list(mps[imps].shape)
                ltensor = environ.read("L", imps - 1)
                if imps == self.site_num - 1:
                    if loop == 1:
                        # the coefficient site
                        rtensor = ones((1, 1, 1))
                        hop = hop_expr(ltensor, rtensor, [mpo[imps]], shape)

                        S_inv = xp.diag(xp.ones(1,dtype=mps.dtype))
                        def func1(y):
                            func = integrand_func_factory(shape, hop, True, S_inv, True,
                                    coef, ovlp_inv1=S_L_inv_list[imps+1],
                                    ovlp_inv0=S_L_inv_list[imps], ovlp0=S_L_list[imps])
                            return func(0, y)

                        ms, Lanczos_vectors = expm_krylov(func1, evolve_dt, mps[imps].ravel().array)
                        logger.debug(f"# of Lanczos_vectors, {Lanczos_vectors}")
                        mps[imps] = ms.reshape(shape)

                    if loop == 1 and self.evolve_config.tdvp_cmf_c_trapz:
                        break
                    else:
                        continue

                # perform qr on the environment mps
                qnbigl, qnbigr, _ = environ_mps._get_big_qn([imps + 1])
                u, s, qnlset, v, s, qnrset = svd_qn.svd_qn(
                    environ_mps[imps + 1].array,
                    qnbigl,
                    qnbigr,
                    environ_mps.qntot,
                    system="R",
                    full_matrices=False,
                )
                vt = v.T

                environ_mps[imps + 1] = vt.reshape(environ_mps[imps + 1].shape)

                rtensor = environ.GetLR(
                    "R", imps + 1, environ_mps, mpo, itensor=None, method="System"
                )

                regular_s = _mu_regularize(s, epsilon=self.evolve_config.reg_epsilon)

                us = u.dot(np.diag(s))

                rtensor = tensordot(rtensor, us, axes=(-1, -1))

                environ_mps[imps] = tensordot(environ_mps[imps], us, axes=(-1, 0))
                environ_mps.qn[imps + 1] = qnrset
                environ_mps.qnidx = imps

                S_inv = u.conj().dot(np.diag(1.0 / regular_s)).T

                hop = hop_expr(ltensor, rtensor, [mpo[imps]], shape)
                func = integrand_func_factory(shape, hop, False, S_inv, True,
                        coef, ovlp_inv1=S_L_inv_list[imps+1],
                        ovlp_inv0=S_L_inv_list[imps], ovlp0=S_L_list[imps])

                sol = solve_ivp(
                    func, (0, evolve_dt), mps[imps].ravel().array, method="RK45"
                )
                cmf_rk_steps.append(len(sol.t))
                ms = sol.y[:, -1].reshape(shape)
                mps[imps] = ms

            if len(cmf_rk_steps) > 0:
                steps_stat = stats.describe(cmf_rk_steps)
                logger.debug(f"{self.evolve_config.method} CMF steps: {steps_stat}")

            if loop == 2:
                environ_mps = mps
                evolve_dt /= 2.
            loop -= 1
            # new_mps.evolve_config.stat = steps_stat

        return mps

    @adaptive_tdvp
    def _evolve_tdvp_ps(self, mpo, evolve_dt) -> "Mps":
        # PhysRevB.94.165116
        # TDVP projector splitting
        # one-site
        if np.iscomplex(evolve_dt):
            mps = self.copy()
        else:
            mps = self.to_complex()

        # construct the environment matrix
        # almost half is not used. Not a big deal.
        environ = Environ(mps, mpo)

        # statistics for debug output
        local_steps = []
        # sweep for 2 rounds
        for i in range(2):
            for imps in mps.iter_idx_list(full=True):
                system = "L" if mps.to_right else "R"
                l_array = environ.read("L", imps - 1)
                r_array = environ.read("R", imps + 1)

                shape = list(mps[imps].shape)
                hop = hop_expr(l_array, r_array, [asxp(mpo[imps].array)], shape)
                mps_t, j = expm_krylov(
                    lambda y: hop(y.reshape(shape)).ravel(),
                    -1j * evolve_dt / 2, mps[imps].ravel().array
                )
                local_steps.append(j)
                mps_t = mps_t.reshape(shape)

                qnbigl, qnbigr, _ = mps._get_big_qn([imps])
                u, qnlset, v, qnrset = svd_qn.svd_qn(
                    asnumpy(mps_t),
                    qnbigl,
                    qnbigr,
                    mps.qntot,
                    QR=True,
                    system=system,
                    full_matrices=False,
                )
                vt = v.T

                if not mps.to_right and imps != 0:
                    mps[imps] = vt.reshape([-1] + shape[1:])
                    mps.qn[imps] = qnrset
                    mps.qnidx = imps-1

                    r_array = environ.GetLR(
                        "R", imps, mps, mpo, itensor=r_array, method="System"
                    )

                    # reverse update u site
                    shape_u = u.shape
                    hop_u = hop_expr(l_array, r_array, [], shape_u)
                    mps_t, j = expm_krylov(
                        lambda y: hop_u(y.reshape(shape_u)).ravel(),
                        1j * evolve_dt / 2, u.ravel()
                    )
                    local_steps.append(j)
                    mps_t = mps_t.reshape(shape_u)

                    mps[imps - 1] = tensordot(mps[imps - 1].array, mps_t, axes=(-1, 0),)

                elif mps.to_right and imps != len(mps) - 1:
                    mps[imps] = u.reshape(shape[:-1] + [-1])
                    mps.qn[imps + 1] = qnlset
                    mps.qnidx = imps+1

                    l_array = environ.GetLR(
                        "L", imps, mps, mpo, itensor=l_array, method="System"
                    )

                    # reverse update svt site
                    shape_svt = vt.shape
                    hop_svt = hop_expr(l_array, r_array, [], shape_svt)
                    mps_t, j = expm_krylov(
                        lambda y: hop_svt(y.reshape(shape_svt)).ravel(),
                        1j * evolve_dt / 2, vt.ravel()
                    )
                    local_steps.append(j)
                    mps_t = mps_t.reshape(shape_svt)

                    mps[imps + 1] = tensordot(mps_t, mps[imps + 1].array, axes=(1, 0),)

                else:
                    mps[imps] = mps_t
            mps._switch_direction()

        steps_stat = stats.describe(local_steps)
        logger.debug(f"TDVP-PS Krylov space: {steps_stat}")
        mps.evolve_config.stat = steps_stat

        return mps

    @adaptive_tdvp
    def _evolve_tdvp_ps2(self, mpo, evolve_dt) -> "Mps":
        # PhysRevB.94.165116
        # TDVP projector splitting
        # two-site
        if np.iscomplex(evolve_dt):
            mps = self.copy()
        else:
            mps = self.to_complex()

        M = self.compress_config.bond_dim_max_value

        # construct the environment matrix
        # almost half is not used. Not a big deal.
        environ = Environ(mps, mpo)

        # statistics for debug output
        local_steps = []
        # sweep for 2 rounds
        for i in range(2):
            for imps in mps.iter_idx_list(full=False):
                if mps.to_right:
                    lidx, cidx0, cidx1, ridx = range(imps - 1, imps + 3)
                    # the idx of the next site
                    cidx2 = cidx1
                    # the idx of the last site
                    last_idx = len(mps) - 2
                else:
                    lidx, cidx0, cidx1, ridx = range(imps - 2, imps + 2)
                    cidx2 = cidx0
                    last_idx = 1

                l_array = environ.read("L", lidx)
                r_array = environ.read("R", ridx)

                # the two-site matrix state
                ms2 = tensordot(mps[cidx0], mps[cidx1], axes=1)
                hop = hop_expr(l_array, r_array, [mpo[cidx0], mpo[cidx1]], ms2.shape)
                mps_t, j = expm_krylov(
                    lambda y: hop(y.reshape(ms2.shape)).ravel(),
                    -1j * evolve_dt / 2,
                    ms2.ravel()
                )
                local_steps.append(j)

                mps_t = mps_t.reshape(ms2.shape)
                qnbigl, qnbigr, _ = mps._get_big_qn([cidx0, cidx1])
                mps._update_mps(mps_t, [cidx0, cidx1], qnbigl, qnbigr, M)
                if mps.compress_config.ofs is not None:
                    mpo.try_swap_site(mps.model, mps.compress_config.ofs_swap_jw)
                if imps == last_idx:
                    continue

                if mps.to_right:
                    l_array = environ.GetLR(
                        "L", lidx + 1, mps, mpo, itensor=l_array, method="System"
                    )
                else:
                    r_array = environ.GetLR(
                        "R", ridx - 1, mps, mpo, itensor=r_array, method="System"
                    )

                # reverse update the next site
                ms1 = mps[cidx2]
                hop = hop_expr(l_array, r_array, [mpo[cidx2]], ms1.shape)
                mps_t, j = expm_krylov(
                    lambda y: hop(y.reshape(ms1.shape)).ravel(),
                    1j * evolve_dt / 2, ms1.ravel()
                )
                local_steps.append(j)
                mps_t = mps_t.reshape(ms1.shape)
                mps[cidx2] = mps_t
                mps._push_cano(cidx2)

            mps._switch_direction()

        steps_stat = stats.describe(local_steps)
        logger.debug(f"TDVP-PS Krylov space: {steps_stat}")
        mps.evolve_config.stat = steps_stat

        return mps

    def evolve_exact(self, h_mpo, evolve_dt, space):
        MPOprop = Mpo.exact_propagator(self.model, -1j * evolve_dt, space, -h_mpo.offset)
        new_mps = MPOprop.apply(self, canonicalise=True)
        self.coeff *= np.exp(-1j * h_mpo.offset * evolve_dt)
        return new_mps

    @property
    def digest(self):
        # used for debugging. Mostly for quickly comparing how two MPSs differ.
        if 10 < self.site_num or self.is_mpdm:
            return None
        prod = np.eye(1).reshape(1, 1, 1)
        for ms in self:
            prod = np.tensordot(prod, ms, axes=1)
            prod = prod.reshape((prod.shape[0], -1, prod.shape[-1]))
        return {"var": prod.var(), "mean": prod.mean(), "ptp": prod.ptp()}

    def todense(self) -> np.array:
        dim = np.prod(self.pbond_list)
        if 20000 < dim:
            raise ValueError("wavefunction too large")
        res = np.ones((1, 1, 1))
        for mt in self:
            dim1 = res.shape[1] * mt.shape[1]
            dim2 = mt.shape[-1]
            res = np.tensordot(res, mt.array, axes=1).reshape(1, dim1, dim2)
        return res[0, :, 0]
    
    def calc_1site_rdm(self, idx=None):
        r""" Calculate 1-site reduced density matrix
        
            :math:`\rho_i = \textrm{Tr}_{j \neq i} | \Psi \rangle \langle \Psi|`
        
        Parameters
        ----------
        idx : int, list, tuple, optional
            site index of 1site_rdm. Default is None, which mean all the rdms
            are calculated.
        
        Returns
        -------
        rdm: Dict
            :math:`\{0:\rho_0, 1:\rho_1, \cdots\}`. The key is the index of the site.
        """

        identity = Mpo.identity(self.model)
        environ = Environ(self, identity, "R")
        if idx is None:
            idx = list(range(self.site_num))
        elif type(idx) is int:
            idx = [idx]
        elif (type(idx) is list) or (type(idx) is tuple):  
            idx = list(idx)
        else:
            assert False

        rdm = {}
        for ims, ms in enumerate(self):
            ltensor = environ.GetLR(
                "L", ims-1, self, identity, itensor=None, method="System"
            )
            rtensor = environ.GetLR(
                "R", ims+1, self, identity, itensor=None, method="Enviro"
            )
            if ims not in idx:
                continue

            ltensor = ltensor.reshape(ltensor.shape[0], ltensor.shape[-1])
            rtensor = rtensor.reshape(rtensor.shape[0], rtensor.shape[-1])
            
            tensor = tensordot(ltensor, ms.conj(), ([0],[0]))
            tensor = tensordot(tensor, rtensor, ([-1],[0]))
            if ms.ndim == 3:
                tensor = tensordot(tensor, ms, ([0,-1],[0,-1]))
            else:
                tensor = tensordot(tensor, ms, ([0,-1,-2],[0,-1,-2]))
            assert xp.allclose(tensor, tensor.T.conj())
            rdm[ims] = asnumpy(tensor)

        return rdm
    
    def calc_2site_rdm(self):
        r""" Calculate 2-site reduced density matrix
        
        :math:`\rho_{ij} = \textrm{Tr}_{k \neq i, k \neq j} | \Psi \rangle \langle \Psi |`.
        
        Returns
        -------
        rdm: Dict
            :math:`\{(0,1):\rho_{01}, (0,2):\rho_{02}, \cdots\}`. The key is a tuple of index of the site.
        """
        
        identity = Mpo.identity(self.model)
        environ_R = Environ(self, identity, "R")
        environ_L = Environ(self, identity, "L")
        L_component = []
        R_component = []
        rdm = {}
        # first construct 1-site environment
        for ims, ms in enumerate(self):
            ltensor = environ_L.GetLR("L", ims-1, self, identity,
                    itensor=None, method="Enviro")
            ltensor = ltensor.reshape(ltensor.shape[0], ltensor.shape[-1])
            tensor = tensordot(ltensor, ms.conj(), ([0],[0]))
            if ms.ndim == 3:
                tensor = tensordot(tensor, ms, ([0],[0]))
            elif ms.ndim == 4:
                tensor = tensordot(tensor, ms, ([0,2],[0,2]))
            L_component.append(tensor.transpose((0,2,1,3)))
            
            rtensor = environ_R.GetLR("R", ims+1, self, identity,
                    itensor=None, method="Enviro")
            rtensor = rtensor.reshape(rtensor.shape[0], rtensor.shape[-1])
            tensor = tensordot(ms.conj(), rtensor, ([-1],[0]))
            if ms.ndim == 3:
                tensor = tensordot(tensor, ms, ([-1],[-1]))
            elif ms.ndim == 4:
                tensor = tensordot(tensor, ms, ([2,-1],[2,-1]))
            R_component.append(tensor.transpose((0,2,1,3)))
        
        # merge two 1-site environment together
        for ims in range(self.site_num):
            tensor = L_component[ims]
            for jms in range(ims+1, self.site_num):
                if jms != ims+1:
                    kms = jms - 1
                    tensor = tensordot(tensor, self[kms].conj(), ([2],[0]))
                    if self[kms].ndim == 3:
                        tensor = tensordot(tensor, self[kms], ([2,3],[0,1]))
                    elif self[kms].ndim == 4:
                        tensor = tensordot(tensor, self[kms], ([2,3,4],[0,1,2]))
                
                rtensor = R_component[jms]
                res = tensordot(tensor, rtensor,
                        ([2,3],[0,1])).transpose(0,2,1,3)
                rdm[(ims, jms)] = asnumpy(res.reshape(res.shape[0]*res.shape[1],-1))
        return rdm
    
    def calc_edof_rdm(self) -> np.ndarray:
        r"""Calculate the reduced density matrix of electronic DoF
        
        :math:`\rho_{ij} = \langle \Psi | a_i^\dagger a_j | \Psi \rangle`
        
        """
        
        key = "edof_reduced_density_matrix"
        n_e = self.model.n_edofs
        e_dofs = self.model.e_dofs
        if key not in self.model.mpos:
            mpos = []
            for idx, dof1 in enumerate(e_dofs):
                for dof2 in e_dofs[idx:]:
                    op = Op(r"a^\dagger a", [dof1, dof2])
                    mpo = Mpo(self.model, terms=op)
                    mpos.append(mpo)
            self.model.mpos[key] = mpos
        else:
            mpos = self.model.mpos[key]
        expectations = deque(self.expectations(mpos))
        reduced_density_matrix = np.zeros((n_e, n_e), dtype=backend.complex_dtype)
        for idx in range(n_e):
            for jdx in range(idx, n_e):
                reduced_density_matrix[idx, jdx] = expectations.popleft()
                reduced_density_matrix[jdx, idx] = np.conj(reduced_density_matrix[idx, jdx])

        return reduced_density_matrix
    
    def calc_entropy(self, entropy_type):
        r""" Calculate 1site, 2site, mutual and bond Von Neumann entropy

            :math:`\textrm{entropy} = -\textrm{Tr}(\rho \ln \rho)`
            where :math:`\ln` stands for natural logarithm.
            
            1site entropy is the entropy between any site and the other
            ``(N-1)`` sites.
            2site entropy is the entropy between any two sites and the other 
            ``(N-2)`` sites.
            mutual entropy characterize the entropy between any two sites.
            bond entropy is the entropy between L-block and R-block.

        Parameters
        ----------
        entropy_type : str
            "1site", "2site", "mutual", "bond"
        
        Returns
        -------
        entropy : dict, ndarray
            if entropy_type = "1site" or "2site", a dictionary is returned and the
            key is the index or the tuple of index of mps sites, else
            an ndarray is returned.
        
        """

        if entropy_type in ["1site", "2site"]:
            if entropy_type == "1site":
                rdm = self.calc_1site_rdm()
            else:
                rdm = self.calc_2site_rdm()
            
            entropy = {}
            for key, dm in rdm.items():
                w, v = scipy.linalg.eigh(dm)
                entropy[key] = calc_vn_entropy(w)

        elif entropy_type == "mutual":
            entropy = self.calc_2site_mutual_entropy()
        elif entropy_type == "bond":
            entropy = self.calc_bond_entropy()
        else:
            raise ValueError(f"unsupported entropy type {entropy_type}")
        return entropy
    
    def calc_2site_mutual_entropy(self):
        r""" 
        Calculate mutual entropy between two sites.
        
        :math:`m_{ij} = (s_i + s_j - s_{ij})/2`
            
        See Chemical Physics 323 (2006) 519–531
        
        Returns
        -------
        mutual_entropy : 2d np.ndarry
            mutual entropy with shape (nsite, nsite)

        """
        entropy_1site = self.calc_entropy("1site")
        entropy_2site = self.calc_entropy("2site")
        nsites = self.site_num
        mut_entropy = np.zeros((nsites, nsites))
        for isite, jsite in itertools.combinations(range(nsites),2):
            key = (isite, jsite) if (isite, jsite) in entropy_2site.keys() else (jsite, isite)
            mut_entropy[isite, jsite] = (entropy_1site[isite] + entropy_1site[jsite] -
                    entropy_2site[key]) / 2
        mut_entropy += mut_entropy.T
        return mut_entropy

    def calc_bond_entropy(self) -> np.ndarray:
        r"""
        Calculate von Neumann entropy at each bond according to :math:`S = -\textrm{Tr}(\rho \ln \rho)`
        where :math:`\rho` is the reduced density matrix of either block.

        Returns
        -------
        S : 1D array
            a NumPy array containing the entropy values.
        
        """
        
        # Make sure that the bond entropy is from the left to the right and not
        # destroy the original mps
        mps = self.copy()
        mps.ensure_right_canonical()
        _, s_list = mps.compress(temp_m_trunc=np.inf, ret_s=True)
        return np.array([calc_vn_entropy(sigma ** 2) for sigma in s_list])

    def dump(self, fname):
        super().dump(fname, other_attrs=["coeff"])

    def __setitem__(self, key, value):
        return super().__setitem__(key, value)

    
    def add(self, other):
        if not np.allclose(self.coeff, other.coeff):
            self.scale(self.coeff, inplace=True)
            other.scale(other.coeff, inplace=True)
            self.coeff = 1
            other.coeff = 1
        return super().add(other)
    
    def distance(self, other) -> float:
        if not np.allclose(self.coeff, other.coeff):
            self.scale(self.coeff, inplace=True)
            other.scale(other.coeff, inplace=True)
            self.coeff = 1
            other.coeff = 1
        return super().distance(other)


def projector(
    ms: xp.ndarray, left: bool, Ovlp_inv1: xp.ndarray = None, Ovlp0: xp.ndarray = None
) -> xp.ndarray:
    if left:
        axes = (-1, -1)
    else:
        axes = (0, 0)

    if Ovlp_inv1 is None:
        proj = xp.tensordot(ms, ms.conj(), axes=axes)
    else:
        # consider the case that the canonical condition is not fulfilled
        if left:
            proj = xp.tensordot(Ovlp0, ms, axes=(-1, 0))
            proj = xp.tensordot(proj, Ovlp_inv1, axes=(-1, 0))
            proj = xp.tensordot(proj, ms.conj(), axes=(-1, -1))
        else:
            proj = xp.tensordot(ms, Ovlp0, axes=(-1, 0))
            proj = xp.tensordot(Ovlp_inv1, proj, axes=(-1, 0))
            proj = xp.tensordot(proj, ms.conj(), axes=(0, 0))

    if left:
        sz = int(np.prod(ms.shape[:-1]))
    else:
        sz = int(np.prod(ms.shape[1:]))
    Iden = xp.array(xp.diag(xp.ones(sz)), dtype=backend.real_dtype).reshape(proj.shape)
    proj = Iden - proj
    return proj

def integrand_func_factory(
    shape,
    hop,
    islast,
    S_inv: Union[np.ndarray, xp.ndarray],
    left: bool,
    coef: complex,
    ovlp_inv1: Union[xp.ndarray, np.ndarray] = None,
    ovlp_inv0: Union[xp.ndarray, np.ndarray] = None,
    ovlp0: Union[xp.ndarray, np.ndarray] = None,
):
    S_inv, ovlp_inv1, ovlp_inv0, ovlp0 = map(asxp, [S_inv, ovlp_inv1, ovlp_inv0, ovlp0])
    # left == True: projector operate on the left side of the HC
    # Ovlp0 is (#.conj, #), Ovlp_inv0 = (#, #.conj), Ovlp_inv1 = (#, #.conj)
    # S_inv is (#.conj, #)
    def func(t, y):
        y0 = asxp(y.reshape(shape))
        HC = hop(y0)
        if not islast:
            proj = projector(y0, left, ovlp_inv1, ovlp0)
            if y0.ndim == 3:
                if left:
                    HC = tensordot(proj, HC, axes=([2, 3], [0, 1]))
                else:
                    HC = tensordot(HC, proj, axes=([1, 2], [2, 3]))
            elif y0.ndim == 4:
                if left:
                    HC = tensordot(proj, HC, axes=([3, 4, 5], [0, 1, 2]))
                else:
                    HC = tensordot(HC, proj, axes=([1, 2, 3], [3, 4, 5]))

        if left:
            if ovlp_inv0 is not None:
                HC = tensordot(ovlp_inv0, HC, axes=(-1, 0))
            return tensordot(HC, S_inv, axes=(-1, 0)).ravel() / coef
        else:
            if ovlp_inv0 is not None:
                HC = tensordot(HC, ovlp_inv0, axes=(-1, -1))
            return tensordot(S_inv, HC, axes=(0, 0)).ravel() / coef

    return func


def transferMat(mps, mpsconj, domain, imps, val) -> np.ndarray:
    """
    calculate the transfer matrix from the left hand or the right hand
    """
    if mpsconj is not None:
        ms, ms_conj = mps[imps].array, mpsconj[imps].array
    else:
        ms = mps[imps].array
        ms_conj = ms.conj()

    if mps[0].ndim == 3:
        if domain == "R":
            val = tensordot(ms_conj, val, axes=(2, 0))
            val = tensordot(val, ms, axes=([1, 2], [1, 2]))
        elif domain == "L":
            val = tensordot(ms_conj, val, axes=(0, 0))
            val = tensordot(val, ms, axes=([0, 2], [1, 0]))
        else:
            assert False
    elif mps[0].ndim == 4:
        if domain == "R":
            val = tensordot(ms_conj, val, axes=(3, 0))
            val = tensordot(val, ms, axes=([1, 2, 3], [1, 2, 3]))
        elif domain == "L":
            val = tensordot(ms_conj, val, axes=(0, 0))
            val = tensordot(val, ms, axes=([0, 3, 1], [1, 0, 2]))
        else:
            assert False
    else:
        raise ValueError(f"the dim of local mps is not correct: {mps[0].ndim}")

    return asnumpy(val)


def _mu_regularize(s, epsilon=1e-10):
    """
    regularization of the singular value of the reduced density matrix
    """
    epsilon = np.sqrt(epsilon)
    return s + epsilon * np.exp(-s / epsilon)


class BraKetPair:
    def __init__(self, bra_mps, ket_mps, mpo=None):
        self.bra_mps = bra_mps
        self.ket_mps = ket_mps
        self.mpo = mpo
        self.ft = self.calc_ft()

    def calc_ft(self):
        if self.mpo is None:
            dot = self.bra_mps.conj().dot(self.ket_mps)
        else:
            dot = self.ket_mps.expectation(self.mpo, self.bra_mps.conj())
        return complex(
            dot * np.conjugate(self.bra_mps.coeff)
            * self.ket_mps.coeff
        )

    def __str__(self):
        if np.iscomplexobj(self.ft):
            # if negative, sign is included in the imag part
            sign = "+" if 0 <= self.ft.imag else ""
            ft_str = "%g%s%gj" % (self.ft.real, sign, self.ft.imag)
        else:
            ft_str = "%g" % self.ft
        return "bra: %s, ket: %s, ft: %s" % (self.bra_mps, self.ket_mps, ft_str)

    def __iter__(self):
        return iter((self.bra_mps, self.ket_mps))


def min_abs(t1, t2):
    # t1, t2 could be int, float, complex
    # return the number with smaller norm

    assert xp.iscomplex(t1) == xp.iscomplex(t2)

    if xp.absolute(t1) < xp.absolute(t2):
        return t1
    else:
        return t2


def _construct_freq_environ(mpos_hash: List[List[int]], hash_to_obj: Dict[int, Matrix], mps: Mps, domain: str, mps_conj):
    """
    Construct environment tensors that are most frequently shown in the group of MPOs
    """
    assert domain in ["L", "R"]
    # count mpo sequence frequency
    counter = Counter()
    for mpo_hash in mpos_hash:
        for i in range(1, len(mpo_hash)+1):
            if domain == "L":
                mpo_seq = mpo_hash[:i]
            else:
                mpo_seq = reversed(mpo_hash[-i:])
            counter.update([tuple(mpo_seq)])

    # transform the counter into a list of matrices.
    # The most frequent sequences first. If the same freq, then shorter sequences first
    # Note that shorter sequences are not less frequent than longer sequences
    most_common = list(counter.items())
    most_common.sort(key=lambda x: (-x[1], len(x[0])))
    matrices_list = []
    hash_list = []
    for hashes, n in most_common:
        # discard unique ones because they do not need to be cached
        if n == 1:
            break
        # cache ``len(mps)`` sequences
        # sequences with the same length may be treated differently.
        if len(mps) < len(matrices_list):
            break
        hash_list.append(hashes)
        matrices_list.append(list(map(hash_to_obj.get, hashes)))

    # contract the tensors
    result = {(): xp.ones((1, 1, 1), dtype=backend.real_dtype)}
    for m_hashes, matrices in zip(hash_list, matrices_list):
        environ = result[tuple(m_hashes[:-1])]
        if domain == "L":
            idx = len(matrices)-1
        else:
            idx = -len(matrices)
        ms, ms_conj = mps[idx], mps_conj[idx]
        result[tuple(m_hashes)] = contract_one_site(environ, ms, matrices[-1], domain=domain, ms_conj=ms_conj)
    return result


def _get_freq_environ(environ_dict, mpo, domain, max_length):
    assert domain in ["L", "R"]

    if domain == "L":
        it = mpo
    else:
        it = reversed(mpo)

    hashes = []
    for mo in it:
        hashes.append(hash(mo))
        if (not tuple(hashes) in environ_dict) or (max_length < len(hashes)):
            hashes.pop()
            break
    if domain == "L":
        i = len(hashes) - 1
    else:
        i = len(mpo) - len(hashes)

    environ = environ_dict[tuple(hashes)]
    return environ, i
