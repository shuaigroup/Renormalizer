# -*- encoding: utf-8 -*-

import logging
from functools import wraps
from typing import Union, List

import numpy as np
import scipy

from scipy import stats


from renormalizer.model import MolList
from renormalizer.lib import solve_ivp, expm_krylov
from renormalizer.mps import svd_qn
from renormalizer.mps.matrix import (
    multi_tensor_contract,
    ones,
    tensordot,
    Matrix,
    asnumpy,)
from renormalizer.mps.backend import backend, xp
from renormalizer.mps.lib import Environ, updatemps, compressed_sum
from renormalizer.mps.mp import MatrixProduct
from renormalizer.mps.mpo import Mpo
from renormalizer.mps.tdh import mflib
from renormalizer.mps.tdh import unitary_propagation
from renormalizer.utils import (
    Quantity,
    OptimizeConfig,
    CompressCriteria,
    CompressConfig,
    EvolveConfig,
    EvolveMethod,
    sizeof_fmt,
)

logger = logging.getLogger(__name__)


def adaptive_tdvp(fun):
    # evolve t/2 (twice) and t to obtain the O(dt^3) error term in 2nd-order Trotter decomposition
    #J. Chem. Phys. 146, 174107 (2017)

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

            mps_half1 = fun(cur_mps, mpo, dt / 2)._dmrg_normalize()
            mps_half2 = fun(mps_half1, mpo, dt / 2)._dmrg_normalize()
            mps = fun(cur_mps, mpo, dt)._dmrg_normalize()
            dis = mps.distance(mps_half2)

            # prevent bug. save "some" memory.
            del mps_half1, mps

            p = (0.75 * config.adaptive_rtol / (dis + 1e-30)) ** (1./3)    
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
    def random(cls, mol_list: MolList, nexciton, m_max, percent=1.0) -> "Mps":
        # a high percent makes the result more random
        # sometimes critical for getting correct optimization result
        mps = cls()
        mps.mol_list = mol_list
        mps.qn = [[0]]
        dim_list = [1]

        for imps in range(len(mol_list.ephtable) - 1):

            # quantum number
            qnbig = np.add.outer(mps.qn[imps], mps._get_sigmaqn(imps)).flatten()
            u_set = []
            s_set = []
            qnset = []

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
            mt, mpsdim, mpsqn, nouse = updatemps(
                u_set, s_set, qnset, u_set, nexciton, m_max, percent=percent
            )
            # add the next mpsdim
            dim_list.append(mpsdim)
            mps.append(
                mt.reshape((dim_list[imps], -1, dim_list[imps + 1]))
            )
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

        # print("self.dim", self.dim)

        mps.tdh_wfns = []
        for mol in mps.mol_list:
            for ph in mol.hartree_phs:
                mps.tdh_wfns.append(np.random.random(ph.n_phys_dim))
        mps.tdh_wfns.append(1.0)

        return mps

    @classmethod
    def gs(cls, mol_list: MolList, max_entangled: bool):
        r"""
        Obtain ground state at :math:`T = 0` or :math:`T = \infty` (maximum entangled).
        Electronic DOFs are always at ground state. and vibrational DOFs depend on ``max_entangled``.
        For Spin-Boson model the electronic DOF also depends on ``max_entangled``.

        Args:
            mol_list (:class:`~renormalizer.model.MolList`): system information.
            max_entanggled (bool): temperature of the vibrational DOFs. If set to ``True``,
                :math:`T = \infty` and if set to ``False``, :math:`T = 0`.
        """
        mps = cls()
        mps.mol_list = mol_list
        mps.qn = [[0]] * (len(mps.ephtable) + 1)
        mps.qnidx = len(mps.ephtable) - 1
        mps.to_right = False
        mps.qntot = 0

        for imol, mol in enumerate(mol_list):
            # electron mps
            if 0 < mol_list.scheme < 4:
                if mol.sbm and max_entangled:
                    array = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
                else:
                    array = np.array([1, 0])
                mps.append(array.reshape((1, 2, 1)))
            elif mol_list.scheme == 4:
                assert not mol.sbm
                if imol == mol_list.mol_num // 2:
                    array = np.zeros(mol_list.mol_num + 1)
                    array[0] = 1
                    mps.append(array.reshape((1, -1, 1)))
            else:
                assert False
            # ph mps
            for ph in mol.dmrg_phs:
                for iboson in range(ph.nqboson):
                    ms = np.zeros((1, ph.base, 1))
                    if max_entangled:
                        ms[0, :, 0] = 1.0 / np.sqrt(ph.base)
                    else:
                        ms[0, 0, 0] = 1.0
                    mps.append(ms)

        mps.tdh_wfns = []

        for mol in mol_list:
            for ph in mol.hartree_phs:
                if max_entangled:
                    diag_elems = [1.0] * ph.n_phys_dim
                    mps.tdh_wfns.append(np.diag(diag_elems))
                else:
                    diag_elems = [1.0] + [0.0] * (ph.n_phys_dim - 1)
                    mps.tdh_wfns.append(np.array(diag_elems))
        # the coefficent a
        mps.tdh_wfns.append(1.0)

        mflib.normalize(mps.tdh_wfns, 1.0)

        return mps

    @classmethod
    def load(cls, mol_list: MolList, fname: str):
        npload = np.load(fname, allow_pickle=True)
        mp = cls()
        mp.mol_list = mol_list
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
        if npload["version"] == "0.1":
            mp.to_right = bool(npload["left"])
            logger.warning("Using old dump/load protocol. TD Hartree part will be lost")
        else:
            mp.to_right = bool(npload["to_right"])
            mp.tdh_wfns = npload["tdh_wfns"]
        return mp

    def __init__(self):
        super(Mps, self).__init__()
        # todo: tdh part with GPU backend
        # tdh part will merge into tdvp evolution scheme in the future
        self.tdh_wfns = [1]

        self.optimize_config: OptimizeConfig = OptimizeConfig()
        self.evolve_config: EvolveConfig = EvolveConfig()

    def conj(self) -> "Mps":
        new_mps = super().conj()
        for idx, wfn in enumerate(new_mps.tdh_wfns):
            new_mps.tdh_wfns[idx] = np.conj(wfn)
        return new_mps

    def dot(self, other: "Mps", with_hartree=True):
        e = super(Mps, self).dot(other)
        if with_hartree:
            assert len(self.tdh_wfns) == len(other.tdh_wfns)
            for wfn1, wfn2 in zip(self.tdh_wfns[:-1], other.tdh_wfns[:-1]):
                # using vdot is buggy here, because vdot will take conjugation automatically
                e *= np.dot(wfn1, wfn2)
        return e

    def to_complex(self, inplace=False) -> "Mps":
        new_mp = super(Mps, self).to_complex(inplace=inplace)
        new_mp.tdh_wfns = [wfn.astype(np.complex128) for wfn in new_mp.tdh_wfns[:-1]] + [
            new_mp.tdh_wfns[-1]
        ]
        return new_mp

    def _get_sigmaqn(self, idx):
        if self.ephtable.is_phonon(idx):
            return [0] * self.pbond_list[idx]
        if self.mol_list.scheme < 4 and self.ephtable.is_electron(idx):
            return [0, 1]
        elif self.mol_list.scheme == 4 and self.ephtable.is_electrons(idx):
            return [0] + [1] * (self.pbond_list[idx] - 1)
        else:
            assert False

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
    def coeff(self):
        return self.tdh_wfns[-1]

    @property
    def hybrid_tdh(self):
        return not self.mol_list.pure_dmrg

    @property
    def nexciton(self):
        return self.qntot

    @property
    def norm(self):
        # return self.dmrg_norm * self.hartree_norm
        return self.tdh_wfns[-1]

    @property
    def dmrg_norm(self) -> float:
        # the fast version in the comment rarely makes sense because in a lot of cases
        # the mps is not canonicalised (though qnidx is set)
        """
        if self.is_left_canon:
            assert self.check_left_canonical()
            return np.linalg.norm(np.ravel(self[-1]))
        else:
            assert self.check_right_canonical()
            return np.linalg.norm(np.ravel(self[0]))
        """
        res = np.sqrt(self.conj().dot(self, with_hartree=False).real)
        return float(res.real)

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
            ret_float = True
        else:
            ret_float = False
        environ = Environ(self, mpo, "R", mps_conj=self_conj)
        l = ones((1, 1, 1))
        r = environ.read("R", 1)
        path = self._expectation_path()
        val = multi_tensor_contract(path, l, self[0], mpo[0], self_conj[0], r).array
        if ret_float:
            return float(val.real)
        else:
            return complex(val)
        # This is time and memory consuming
        # return self_conj.dot(mpo.apply(self), with_hartree=False).real

    def expectations(self, mpos, opt=True) -> np.ndarray:
        if not opt:
            return np.array([self.expectation(mpo) for mpo in mpos])
        else:
            # only supports local operator now
            # id can be used as efficient hash because of `Matrix` implementation
            mpo_ids = np.array([[id(m) for m in mpo] for mpo in mpos])
            common_mpo_ids = mpo_ids[0].copy()
            mpo0_unique_idx = np.where(np.sum(mpo_ids == common_mpo_ids, axis=0) == 1)[0][0]
            common_mpo_ids[mpo0_unique_idx] = mpo_ids[1][mpo0_unique_idx]
            x, unique_idx = np.where(mpo_ids != common_mpo_ids)
            # should find one at each line
            assert xp.allclose(x, np.arange(len(mpos)))
            common_mpo = list(mpos[0])
            common_mpo[mpo0_unique_idx] = mpos[1][mpo0_unique_idx]
            self_conj = self._expectation_conj()
            environ = Environ(self, common_mpo, mps_conj=self_conj)
            res_list = []
            for idx, mpo in zip(unique_idx, mpos):
                l = environ.read("L", idx - 1)
                r = environ.read("R", idx + 1)
                path = self._expectation_path()
                res = multi_tensor_contract(path, l, self[idx], mpo[idx], self_conj[idx], r)
                res_list.append(float(res.real))
            return np.array(res_list)
            # the naive way, slow and time consuming
            # return np.array([self.expectation(mpo) for mpo in mpos])

    @property
    def ph_occupations(self):
        key = "ph_occupations"
        if key not in self.mol_list.mpos:
            mpos = []
            for imol, mol in enumerate(self.mol_list):
                for iph in range(len(mol.dmrg_phs)):
                    mpos.append(Mpo.ph_onsite(self.mol_list, r"b^\dagger b", imol, iph))
            self.mol_list.mpos[key] = mpos
        else:
            mpos = self.mol_list.mpos[key]
        return self.expectations(mpos)

    @property
    def e_occupations(self):
        if self.mol_list.scheme < 4:
            key = "e_occupations"
            if key not in self.mol_list.mpos:
                mpos = [
                    Mpo.onsite(self.mol_list, r"a^\dagger a", mol_idx_set={i})
                    for i in range(self.mol_num)
                ]
                self.mol_list.mpos[key] = mpos
            else:
                mpos = self.mol_list.mpos[key]
            return self.expectations(mpos)
        elif self.mol_list.scheme == 4:
            # get rdm is very fast
            rdm = self.calc_reduced_density_matrix()
            return np.diag(rdm).real
        else:
            assert False

    def metacopy(self) -> "Mps":
        new = super().metacopy()
        new.tdh_wfns = [wfn.copy() for wfn in self.tdh_wfns[:-1]] + [self.tdh_wfns[-1]]
        new.optimize_config = self.optimize_config
        # evolve_config has its own data
        new.evolve_config = self.evolve_config.copy()
        return new

    def _dmrg_normalize(self):
        return self.scale(1.0 / self.dmrg_norm, inplace=True)

    def normalize(self, norm=None):
        # real time propagation: dmrg should be normalized, tdh should be normalized, coefficient is not changed,
        #  use norm=None
        # imag time propagation: dmrg should be normalized, tdh should be normalized, coefficient is normalized to 1.0
        # applied by a operator then normalize: dmrg should be normalized,
        #   tdh should be normalized, coefficient is set to the length
        # these two cases should set `norm` equals to corresponding value
        self._dmrg_normalize()
        if norm is None:
            mflib.normalize(self.tdh_wfns, self.tdh_wfns[-1])
        else:
            mflib.normalize(self.tdh_wfns, norm)
        return self

    def canonical_normalize(self):
        # applied by a operator then normalize: dmrg should be normalized,
        #   tdh should be normalized, coefficient is set to the length
        # suppose length is only determined by dmrg part
        return self.normalize(self.dmrg_norm)

    def expand_bond_dimension(self, hint_mpo=None, coef=1e-10):
        """
        expand bond dimension as required in compress_config
        """
        if not self.use_dummy_qn and self.nexciton == 0:
            raise ValueError("Expanding bond dimensional without exciton is meaningless")
        # expander m target
        m_target = self.compress_config.bond_dim_max_value - self.bond_dims_mean
        # will be restored at exit
        self.compress_config.bond_dim_max_value = m_target
        if self.compress_config.criteria is not CompressCriteria.fixed:
            logger.warning("Setting compress criteria to fixed")
            self.compress_config.criteria = CompressCriteria.fixed
        logger.debug(f"target for expander: {m_target}")
        if hint_mpo is None:
            expander = self.__class__.random(self.mol_list, 1, m_target)
        else:
            # fill states related to `hint_mpo`
            logger.debug(f"average bond dimension of hint mpo: {hint_mpo.bond_dims_mean}")
            # in case of localized `self`
            if not self.use_dummy_qn:
                if self.is_mps:
                    ex_state: MatrixProduct = self.random(self.mol_list, 1, 10)
                elif self.is_mpdm:
                    ex_state: MatrixProduct = self.max_entangled_ex(self.mol_list)
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
                logger.debug(f"cumulated bond dimension: {cumulated_m}. lastone bond dimension: {lastone.bond_dims}")
                if m_target < cumulated_m:
                    break
                if m_target < 0.8 * (lastone.bond_dims_mean * hint_mpo.bond_dims_mean):
                    lastone = lastone.canonicalise().compress(m_target // hint_mpo.bond_dims_mean)
                lastone = hint_mpo @ lastone
        logger.debug(f"expander bond dimension: {expander.bond_dims}")
        self.compress_config.bond_dim_max_value += self.bond_dims_mean
        return (self + expander.scale(coef, inplace=True)).canonicalise().canonicalise().canonical_normalize()

    def evolve(self, mpo, evolve_dt):
        if self.hybrid_tdh:
            hybrid_mpo, HAM, Etot = self.construct_hybrid_Ham(mpo)
            mps = self.evolve_dmrg(hybrid_mpo, evolve_dt)
            unitary_propagation(mps.tdh_wfns, HAM, Etot, evolve_dt)
        else:
            # save the cost of calculating energy
            mps = self.evolve_dmrg(mpo, evolve_dt)
        if np.iscomplex(evolve_dt):
            mps.normalize(1.0)
        else:
            mps.normalize(None)
        return mps

    def evolve_dmrg(self, mpo, evolve_dt) -> "Mps":

        method = {
            EvolveMethod.prop_and_compress: self._evolve_dmrg_prop_and_compress,
            EvolveMethod.tdvp_mu_vmf: self._evolve_dmrg_tdvp_mu_vmf,
            EvolveMethod.tdvp_vmf: self._evolve_dmrg_tdvp_mu_vmf,
            EvolveMethod.tdvp_mu_cmf: self._evolve_dmrg_tdvp_mu_cmf,
            EvolveMethod.tdvp_ps: self._evolve_dmrg_tdvp_ps,
        }[self.evolve_config.method]
        new_mps = method(mpo, evolve_dt)
        return new_mps

    def _evolve_dmrg_prop_and_compress(self, mpo, evolve_dt) -> "Mps":
        """
        The global propagation & compression evolution scheme
        """
        config = self.evolve_config
        assert evolve_dt is not None

        propagation_c = config.rk_config.coeff
        termlist = [self]
        # don't let bond dim grow when contracting
        orig_compress_config = self.compress_config
        contract_compress_config = self.compress_config.copy()
        if contract_compress_config.criteria is CompressCriteria.threshold:
            contract_compress_config.criteria = CompressCriteria.both
        #contract_compress_config.min_dims = None
        #contract_compress_config.max_dims = np.array(self.bond_dims) + 4
        self.compress_config = contract_compress_config

        while len(termlist) < len(propagation_c):
            termlist.append(mpo.contract(termlist[-1]))
        # bond dim can grow after adding
        for t in termlist:
            t.compress_config = orig_compress_config

        if config.adaptive:
            config.check_valid_dt(evolve_dt)
            
            p_restart = 0.5 # restart threshold 
            p_min = 0.1     # safeguard for minimal allowed p
            p_max = 2.      # safeguard for maximal allowed p
            
            while True:
                scaled_termlist = []
                dt = min_abs(config.guess_dt, evolve_dt)
                logger.debug(
                        f"guess_dt: {config.guess_dt}, try time step size: {dt}"
                )
                for idx, term in enumerate(termlist):
                    scale = (-1.0j * dt) ** idx * propagation_c[idx]
                    scaled_termlist.append(term.scale(scale))
                del term
                new_mps1 = compressed_sum(scaled_termlist[:-1])._dmrg_normalize()
                new_mps2 = compressed_sum([new_mps1, scaled_termlist[-1]])._dmrg_normalize()
                dis = new_mps1.distance(new_mps2)
                # 0.2 is 1/5 for RK45
                p = (config.adaptive_rtol / (dis + 1e-30)) ** 0.2    
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
                        new_mps2.evolve_config.guess_dt = min_abs(dt*p, config.guess_dt)
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
                        del new_mps1, termlist, scaled_termlist  # memory consuming and not useful anymore
                        logger.debug(
                            f"evolution converged, new guess_dt: {config.guess_dt}"
                        )
                        logger.debug(f"sub-step {dt} further, remaining: {new_dt}")
                        return new_mps2._evolve_dmrg_prop_and_compress(mpo, new_dt)
        else:
            for idx, term in enumerate(termlist):
                term.scale(
                    (-1.0j * evolve_dt) ** idx * propagation_c[idx], inplace=True
                )
            return compressed_sum(termlist)

    def _evolve_dmrg_tdvp_mu_vmf(self, mpo, evolve_dt) -> "Mps":
        """
        variable mean field 
        see the difference between VMF and CMF, refer to Z. Phys. D 42, 113–129 (1997)
        the matrix unfolding algorithm, see arXiv:1907.12044 
        only the RKF45 integration is used.
        The default RKF45 local step error tolerance is rtol:1e-5, atol:1e-8
        regulation of S is 1e-10, these default parameters could be changed in
        /utils/configs.py

        """

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
            self.ensure_left_canon()

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
            qnbigl, qnbigr, qnmat= mps._get_big_qn(imps)
            qnmat_list.append(qnmat)
            position.append(position[-1]+np.sum(qnmat == qntot))

        sw_min_list = []

        def func_vmf(t,y):
            
            sw_min_list.clear()

            # update mps: from left to right
            for imps in range(mps.site_num):
                mps[imps] = svd_qn.cvec2cmat(mps[imps].shape, y[position[imps]:position[imps+1]],
                        qnmat_list[imps], qntot)
            
            if self.evolve_config.method == EvolveMethod.tdvp_mu_vmf:
                environ_mps = mps.copy()
            elif self.evolve_config.method == EvolveMethod.tdvp_vmf:
                environ_mps = mps
                # the first S_R
                S_R = ones([1, 1], dtype=mps.dtype)
            else:
                assert False

            environ = Environ(environ_mps, mpo, "L")
            environ.write_r_sentinel(environ_mps)
            
            if self.evolve_config.force_ovlp:
                # construct the S_L list (type: Matrix) and S_L_inv list (type: xp.array)
                # len: mps.site_num+1
                S_L_list = [ones([1, 1], dtype=mps.dtype),]
                for imps in range(mps.site_num):
                    S_L_list.append(transferMat(mps, mps.conj(), "L", imps,
                        S_L_list[imps]))
                
                S_L_inv_list = []    
                for imps in range(mps.site_num+1):
                    w, u = scipy.linalg.eigh(S_L_list[imps].asnumpy())
                    S_L_inv = xp.asarray(u.dot(np.diag(1.0 / w)).dot(u.T.conj()))
                    S_L_inv_list.append(S_L_inv)
                    S_L_list[imps] = S_L_list[imps].array
            else:
                S_L_list = [None,] * (mps.site_num+1)
                S_L_inv_list = [None,] * (mps.site_num+1)
            
            # calculate hop_y: from right to left
            hop_y = xp.empty_like(y)

            for imps in mps.iter_idx_list(full=True):
                shape = list(mps[imps].shape)
                ltensor = environ.read("L", imps - 1)
                
                if imps == self.site_num - 1:
                    # the coefficient site
                    rtensor = ones((1, 1, 1))
                    hop = hop_factory(ltensor, rtensor, mpo[imps], len(shape))
                    
                    S_inv = xp.diag(xp.ones(1,dtype=mps.dtype))
                    func = integrand_func_factory(shape, hop, True, S_inv, True,
                            coef, Ovlp_inv1=S_L_inv_list[imps+1],
                            Ovlp_inv0=S_L_inv_list[imps], Ovlp0=S_L_list[imps])
                               
                    hop_y[position[imps]:position[imps+1]] = func(0,
                            mps[imps].array.ravel()).reshape(mps[imps].shape)[qnmat_list[imps]==qntot]

                    continue
                
                if self.evolve_config.method == EvolveMethod.tdvp_mu_vmf:
                    # perform qr on the environment mps
                    qnbigl, qnbigr, _ = environ_mps._get_big_qn(imps + 1)
                    u, s, qnlset, v, s, qnrset = svd_qn.Csvd(
                            environ_mps[imps + 1].asnumpy(), qnbigl, qnbigr,
                            environ_mps.qntot, system="R", full_matrices=False)
                    vt = v.T

                    environ_mps[imps + 1] = vt.reshape(environ_mps[imps + 1].shape)
                
                    rtensor = environ.GetLR(
                        "R", imps + 1, environ_mps, mpo, itensor=None, method="System"
                    )
                    
                    sw_min_list.append(s.min())
                    regular_s = _mu_regularize(s, epsilon=self.evolve_config.reg_epsilon)
                    
                    u = xp.asarray(u)
                    us = Matrix(u.dot(xp.diag(s)))

                    rtensor = tensordot(rtensor, us, axes=(-1, -1))
                    
                    environ_mps[imps] = tensordot(environ_mps[imps], us, axes=(-1, 0))
                    environ_mps.qn[imps + 1] = qnrset
                    environ_mps.qnidx = imps

                    S_inv = u.conj().dot(xp.diag(1.0 / regular_s)).T
                
                elif self.evolve_config.method == EvolveMethod.tdvp_vmf:
                    rtensor = environ.GetLR(
                        "R", imps + 1, environ_mps, mpo, itensor=None, method="System")
                    
                    # regularize density matrix
                    # Note that S_R is (#.conj, #)
                    S_R = transferMat(environ_mps, environ_mps.conj(), "R", imps + 1, Matrix(S_R)).asnumpy()
                    w, u = scipy.linalg.eigh(S_R)
                    
                    # discard the negative eigenvalues due to numerical error
                    w = np.where(w>0, w, 0)
                    
                    sw_min_list.append(w.min())

                    epsilon = self.evolve_config.reg_epsilon
                    w = w + epsilon * np.exp(-w / epsilon)
                    
                    u = xp.asarray(u)
                    # S_inv is (#.conj, #)
                    S_inv = u.dot(xp.diag(1.0 / w)).dot(u.T.conj()).T

                hop = hop_factory(ltensor, rtensor, mpo[imps], len(shape))

                func = integrand_func_factory(shape, hop, False, S_inv, True,
                        coef, Ovlp_inv1=S_L_inv_list[imps+1],
                        Ovlp_inv0=S_L_inv_list[imps], Ovlp0=S_L_list[imps])
                
                hop_y[position[imps]:position[imps+1]] = func(0,
                        mps[imps].array.ravel()).reshape(mps[imps].shape)[qnmat_list[imps]==qntot]
            
            return hop_y

        init_y = xp.concatenate([ms.array[qnmat_list[ims]==qntot] for ims, ms in enumerate(mps)])
        # the ivp local error, please refer to the Scipy default setting
        sol = solve_ivp( func_vmf, (0, evolve_dt), init_y, method="RK45",
                rtol=self.evolve_config.ivp_rtol,
                atol=self.evolve_config.ivp_atol)
        
        # update mps: from left to right
        for imps in range(mps.site_num):
            mps[imps] = svd_qn.cvec2cmat(mps[imps].shape, sol.y[:,-1][position[imps]:position[imps+1]],
                    qnmat_list[imps], qntot)
        
        logger.debug(f"{self.evolve_config.method} VMF func called: {sol.nfev}. RKF steps: {len(sol.t)}")
        
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
    def _evolve_dmrg_tdvp_mu_cmf(self, mpo, evolve_dt) -> "Mps":
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
        
        self.ensure_left_canon()

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
            environ_mps = self.evolve_dmrg(mpo, evolve_dt / 2)
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
            environ.write_r_sentinel(environ_mps)

            # statistics for debug output
            cmf_rk_steps = []
            
            if self.evolve_config.force_ovlp:
                # construct the S_L list (type: Matrix) and S_L_inv list (type: xp.array)
                # len: mps.site_num+1
                S_L_list = [ones([1, 1], dtype=mps.dtype),]
                for imps in range(mps.site_num):
                    S_L_list.append(transferMat(environ_mps, environ_mps.conj(), "L", imps,
                        S_L_list[imps]))
                
                S_L_inv_list = []    
                for imps in range(mps.site_num+1):
                    w, u = scipy.linalg.eigh(S_L_list[imps].asnumpy())
                    S_L_inv = xp.asarray(u.dot(np.diag(1.0 / w)).dot(u.T.conj()))
                    S_L_inv_list.append(S_L_inv)
                    S_L_list[imps] = S_L_list[imps].array
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
                        hop = hop_factory(ltensor, rtensor, mpo[imps], len(shape))

                        S_inv = xp.diag(xp.ones(1,dtype=mps.dtype))
                        def func1(y):
                            func = integrand_func_factory(shape, hop, True, S_inv, True,
                                    coef, Ovlp_inv1=S_L_inv_list[imps+1],
                                    Ovlp_inv0=S_L_inv_list[imps], Ovlp0=S_L_list[imps])
                            return func(0, y)

                        ms, Lanczos_vectors = expm_krylov(func1, evolve_dt, mps[imps].ravel().array)
                        logger.debug(f"# of Lanczos_vectors, {Lanczos_vectors}")
                        mps[imps] = ms.reshape(shape)
                    
                    if loop == 1 and self.evolve_config.tdvp_cmf_c_trapz:
                        break
                    else:
                        continue

                # perform qr on the environment mps
                qnbigl, qnbigr, _ = environ_mps._get_big_qn(imps + 1)
                u, s, qnlset, v, s, qnrset = svd_qn.Csvd(
                    environ_mps[imps + 1].asnumpy(),
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

                us = Matrix(u.dot(np.diag(s)))

                rtensor = tensordot(rtensor, us, axes=(-1, -1))

                environ_mps[imps] = tensordot(environ_mps[imps], us, axes=(-1, 0))
                environ_mps.qn[imps + 1] = qnrset
                environ_mps.qnidx = imps

                S_inv = Matrix(u).conj().dot(xp.diag(1.0 / regular_s)).T

                hop = hop_factory(ltensor, rtensor, mpo[imps], len(shape))
                func = integrand_func_factory(shape, hop, False, S_inv.array, True,
                        coef, Ovlp_inv1=S_L_inv_list[imps+1],
                        Ovlp_inv0=S_L_inv_list[imps], Ovlp0=S_L_list[imps])

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
    def _evolve_dmrg_tdvp_ps(self, mpo, evolve_dt) -> "Mps":
        # PhysRevB.94.165116
        # TDVP projector splitting
        imag_time = np.iscomplex(evolve_dt)
        if imag_time:
            mps = self.copy()
            mps_conj = mps
        else:
            mps = self.to_complex()
            mps_conj = mps.conj()  # another copy, so 3x memory is used.

        # construct the environment matrix
        # almost half is not used. Not a big deal.
        environ = Environ(mps, mpo)

        # a workaround for https://github.com/scipy/scipy/issues/10164
        if imag_time:
            evolve_dt = -evolve_dt.imag
            # used in calculating derivatives
            coef = -1
        else:
            coef = 1j

        # todo: remove USE_RK if proved to be useless
        USE_RK = False
        # statistics for debug output
        local_steps = []
        # sweep for 2 rounds
        for i in range(2):
            for imps in mps.iter_idx_list(full=True):
                system = "L" if mps.to_right else "R"
                ltensor = environ.read("L", imps - 1)
                rtensor = environ.read("R", imps + 1)

                shape = list(mps[imps].shape)
                l_array = ltensor.array
                r_array = rtensor.array

                hop = hop_factory(l_array, r_array, mpo[imps].array, len(shape))

                def hop_svt(ms):
                    # S-a   l-S
                    #
                    # O-b - b-O
                    #
                    # S-c   k-S

                    path = [([0, 1], "abc, ck -> abk"), ([1, 0], "abk, lbk -> al")]
                    HC = multi_tensor_contract(path, l_array, ms, r_array)
                    return HC

                if USE_RK:
                    def func(t, y):
                        return hop(y.reshape(shape)).ravel() / coef
                    sol = solve_ivp(
                        func, (0, evolve_dt / 2.0), mps[imps].ravel().array, method="RK45"
                    )
                    local_steps.append(len(sol.t))
                    mps_t = sol.y[:, -1]
                else:
                    # Can't use the same func because here H should be Hermitian
                    def func(y):
                        return hop(y.reshape(shape)).ravel()
                    mps_t, j = expm_krylov(func, (evolve_dt / 2) / coef, mps[imps].ravel().array)
                    local_steps.append(j)
                mps_t = mps_t.reshape(shape)

                qnbigl, qnbigr, _ = mps._get_big_qn(imps)
                u, qnlset, v, qnrset = svd_qn.Csvd(
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
                    mps_conj[imps] = mps[imps].conj()
                    mps.qn[imps] = qnrset
                    mps.qnidx = imps-1

                    rtensor = environ.GetLR(
                        "R", imps, mps, mpo, itensor=rtensor, method="System"
                    )
                    r_array = rtensor.array

                    # reverse update u site
                    shape_u = u.shape

                    if USE_RK:
                        def func_u(t, y):
                            return hop_svt(y.reshape(shape_u)).ravel() / coef
                        sol_u = solve_ivp(
                            func_u, (0, -evolve_dt / 2), u.ravel(), method="RK45"
                        )
                        local_steps.append(len(sol_u.t))
                        mps_t = sol_u.y[:, -1]
                    else:
                        def func_u(y):
                            return hop_svt(y.reshape(shape_u)).ravel()
                        mps_t, j = expm_krylov(func_u, (-evolve_dt / 2) / coef, u.ravel())
                        local_steps.append(j)
                    mps_t = mps_t.reshape(shape_u)

                    mps[imps - 1] = tensordot(
                        mps[imps - 1].array,
                        mps_t,
                        axes=(-1, 0),
                    )
                    mps_conj[imps - 1] = mps[imps - 1].conj()

                elif mps.to_right and imps != len(mps) - 1:
                    mps[imps] = u.reshape(shape[:-1] + [-1])
                    mps_conj[imps] = mps[imps].conj()
                    mps.qn[imps + 1] = qnlset
                    mps.qnidx = imps+1

                    ltensor = environ.GetLR(
                        "L", imps, mps, mpo, itensor=ltensor, method="System"
                    )
                    l_array = ltensor.array

                    # reverse update svt site
                    shape_svt = vt.shape

                    if USE_RK:
                        def func_svt(t, y):
                            return hop_svt(y.reshape(shape_svt)).ravel() / coef
                        sol_svt = solve_ivp(
                            func_svt, (0, -evolve_dt / 2), vt.ravel(), method="RK45"
                        )
                        local_steps.append(len(sol_svt.t))
                        mps_t = sol_svt.y[:, -1]
                    else:
                        def func_svt(y):
                            return hop_svt(y.reshape(shape_svt)).ravel()
                        mps_t, j = expm_krylov(func_svt, (-evolve_dt / 2) / coef, vt.ravel())
                        local_steps.append(j)
                    mps_t = mps_t.reshape(shape_svt)

                    mps[imps + 1] = tensordot(
                        mps_t,
                        mps[imps + 1].array,
                        axes=(1, 0),
                    )
                    mps_conj[imps + 1] = mps[imps + 1].conj()

                else:
                    mps[imps] = mps_t
                    mps_conj[imps] = mps[imps].conj()
            mps._switch_direction()

        steps_stat = stats.describe(local_steps)
        logger.debug(f"TDVP-PS CMF steps: {steps_stat}")
        mps.evolve_config.stat = steps_stat

        return mps

    def evolve_exact(self, h_mpo, evolve_dt, space):
        MPOprop, HAM, Etot = self.hybrid_exact_propagator(
            h_mpo, -1.0j * evolve_dt, space
        )
        new_mps = MPOprop.apply(self, canonicalise=True)
        unitary_propagation(new_mps.tdh_wfns, HAM, Etot, evolve_dt)
        return new_mps

    @property
    def digest(self):
        if 10 < self.site_num or self.is_mpdm:
            return None
        prod = np.eye(1).reshape(1, 1, 1)
        for ms in self:
            prod = np.tensordot(prod, ms, axes=1)
            prod = prod.reshape((prod.shape[0], -1, prod.shape[-1]))
        return {"var": prod.var(), "mean": prod.mean(), "ptp": prod.ptp()}

    # put the below 2 constructors here because they really depend on the implement details of MPS (at least the
    # Hartree part).
    def construct_hybrid_Ham(self, mpo_indep, debug=False):
        """
        construct hybrid DMRG and Hartree(-Fock) Hamiltonian
        """
        mol_list = mpo_indep.mol_list
        WFN = self.tdh_wfns
        nmols = len(mol_list)

        # many-body electronic part
        A_el = self.e_occupations

        logger.debug("dmrg_occ: %s" % A_el)

        # many-body vibration part
        B_vib = []
        iwfn = 0
        for imol in range(nmols):
            B_vib.append([])
            for ph in mol_list[imol].hartree_phs:
                B_vib[imol].append(mflib.exp_value(WFN[iwfn], ph.h_dep, WFN[iwfn]))
                iwfn += 1
        B_vib_mol = [np.sum(np.array(i)) for i in B_vib]

        Etot = 0.0
        # construct new HMPO
        e_mean = self.expectation(mpo_indep)
        elocal_offset = np.array(
            [mol_list[imol].hartree_e0 + B_vib_mol[imol] for imol in range(nmols)]
        ).real
        e_mean += A_el.dot(elocal_offset)
        total_offset = mpo_indep.offset + Quantity(e_mean.real)
        MPO = Mpo(
            mol_list,
            mpo_indep.rep,
            elocal_offset=elocal_offset,
            offset=total_offset,
        )

        Etot += e_mean

        iwfn = 0
        HAM = []
        for imol, mol in enumerate(mol_list):
            for iph, ph in enumerate(mol.hartree_phs):
                e_mean = mflib.exp_value(WFN[iwfn], ph.h_indep, WFN[iwfn])
                Etot += e_mean.real
                e_mean += A_el[imol] * B_vib[imol][iph]
                HAM.append(
                    ph.h_indep
                    + ph.h_dep * A_el[imol]
                    - np.diag([e_mean] * WFN[iwfn].shape[0])
                )
                iwfn += 1
        logger.debug("Etot= %g" % Etot)
        if debug:
            return MPO, HAM, Etot, A_el
        else:
            return MPO, HAM, Etot

    # provide e_mean and mpo_indep separately because e_mean can be precomputed and stored to avoid multiple computation
    def hybrid_exact_propagator(self, mpo_indep, x, space="GS"):
        """
        construct the exact propagator in the GS space or single molecule
        """
        assert space in ["GS", "EX"]

        e_mean = self.expectation(mpo_indep)
        logger.debug("e_mean in exact propagator: %g" % e_mean)

        total_offset = (mpo_indep.offset + Quantity(e_mean.real)).as_au()
        MPOprop = Mpo.exact_propagator(
            self.mol_list, x, space=space, shift=-total_offset
        )

        Etot = total_offset

        # TDH propagator
        iwfn = 0
        HAM = []
        for mol in self.mol_list:
            for ph in mol.hartree_phs:
                h_vib_indep = ph.h_indep
                h_vib_dep = ph.h_dep
                e_mean = mflib.exp_value(self.tdh_wfns[iwfn], h_vib_indep, self.tdh_wfns[iwfn])
                if space == "EX":
                    e_mean += mflib.exp_value(
                        self.tdh_wfns[iwfn], h_vib_dep, self.tdh_wfns[iwfn]
                    )
                Etot += e_mean

                if space == "GS":
                    ham = h_vib_indep - np.diag([e_mean] * ph.n_phys_dim)
                elif space == "EX":
                    ham = h_vib_indep + h_vib_dep - np.diag([e_mean] * ph.n_phys_dim)
                else:
                    assert False

                HAM.append(ham)
                iwfn += 1

        return MPOprop, HAM, Etot

    def hartree_wfn_diff(self, other: "Mps"):
        assert len(self.tdh_wfns) == len(other.tdh_wfns)
        res = []
        for wfn1, wfn2 in zip(self.tdh_wfns, other.tdh_wfns):
            res.append(
                scipy.linalg.norm(
                    np.tensordot(wfn1, wfn1, axes=0) - np.tensordot(wfn2, wfn2, axes=0)
                )
            )
        return np.array(res)

    def full_wfn(self) -> xp.array:
        dim = np.prod(self.pbond_list)
        if 20000 < dim:
            raise ValueError("wavefunction too large")
        res = ones((1, 1, 1))
        for mt in self:
            dim1 = res.shape[1] * mt.shape[1]
            dim2 = mt.shape[-1]
            res = tensordot(res, mt, axes=1).reshape(1, dim1, dim2)
        return res[0, :, 0].asnumpy()

    def _calc_reduced_density_matrix(self, mp1, mp2):
        if self.mol_list.scheme < 4:
            # further optimization is difficult. There are totally N^2 intermediate results to remember.
            reduced_density_matrix = np.zeros(
                (self.mol_list.mol_num, self.mol_list.mol_num), dtype=backend.complex_dtype
            )
            for i in range(self.mol_list.mol_num):
                for j in range(self.mol_list.mol_num):
                    elem = ones((1, 1))
                    e_idx = -1
                    for mt_idx, (mt1, mt2) in enumerate(zip(mp1, mp2)):
                        if self.ephtable.is_electron(mt_idx):
                            e_idx += 1
                            axis_idx1 = int(e_idx == i)
                            axis_idx2 = int(e_idx == j)
                            sub_mt1 = mt1[:, axis_idx1, :, :]
                            sub_mt2 = mt2[:, :, axis_idx2, :]
                            elem = tensordot(elem, sub_mt1, axes=(0, 0))
                            elem = tensordot(elem, sub_mt2, axes=[(0, 1), (0, 1)])
                        else:
                            elem = tensordot(elem, mt1, axes=(0, 0))
                            elem = tensordot(elem, mt2, axes=[(0, 1, 2), (0, 2, 1)])
                    reduced_density_matrix[i][j] = elem.flatten()[0]
        elif self.mol_list.scheme == 4:
            e_idx = self.mol_list.e_idx()
            l = ones((1, 1))
            for i in range(e_idx):
                l = tensordot(l, mp1[i], axes=(0, 0))
                l = tensordot(l, mp2[i], axes=([0, 1, 2], [0, 2, 1]))
            r = ones((1, 1))
            for i in range(len(self)-1, e_idx, -1):
                r = tensordot(mp1[i], r, axes=(3, 0))
                r = tensordot(r, mp2[i], axes=([1, 2, 3], [2, 1, 3]))
            #       f
            #       |
            # S--b--S--g--S
            # |     |     |
            # |     c     |
            # |     |     |
            # S--a--S--e--S
            #       |
            #       d
            path = [
                ([0, 1], "ab, adce -> bdce"),
                ([2, 0], "bdce, bcfg -> defg"),
                ([1, 0], "defg, eg -> df"),
            ]
            reduced_density_matrix = asnumpy(multi_tensor_contract(path, l, mp1[e_idx], mp2[e_idx], r).array)[1:, 1:]
        else:
            assert False
        return reduced_density_matrix

    def calc_reduced_density_matrix(self) -> np.ndarray:
        mp1 = [mt.reshape(mt.shape[0], mt.shape[1], 1, mt.shape[2]) for mt in self]
        mp2 = [mt.reshape(mt.shape[0], 1, mt.shape[1], mt.shape[2]).conj() for mt in self]
        return self._calc_reduced_density_matrix(mp1, mp2)

    def calc_vn_entropy(self) -> np.ndarray:
        r"""
        Calculate von Neumann entropy at each bond according to :math:`S = -\textrm{Tr}(\rho \ln \rho)`
        where :math:`\rho` is the density matrix.

        Returns:
            a NumPy array containing the entropy values.
        """
        _, s_list = self.compress(temp_m_trunc=np.inf, ret_s=True)
        entropy_list = []
        for sigma in s_list:
            rho = sigma ** 2
            normed_rho = rho / rho.sum()
            truncate_rho = normed_rho[0 < normed_rho]
            entropy = - (truncate_rho * np.log(truncate_rho)).sum()
            entropy_list.append(entropy)
        return np.array(entropy_list)

    def dump(self, fname):
        data_dict = dict()
        # version of the protocol
        data_dict["version"] = "0.2"
        data_dict["nsites"] = len(self)
        for idx, mt in enumerate(self):
            data_dict[f"mt_{idx}"] = mt.asnumpy()
        for attr in ["qn", "qnidx", "qntot", "to_right", "tdh_wfns"]:
            data_dict[attr] = getattr(self, attr)
        try:
            np.savez(fname, **data_dict)
        except Exception as e:
            logger.error(f"Dump mps failed, exception info: f{e}")

    def __str__(self):
        template_str = "current size: {}, Matrix product bond dim:{}"
        return template_str.format(
            sizeof_fmt(self.total_bytes),
            self.bond_dims,
        )

    def __setitem__(self, key, value):
        return super().__setitem__(key, value)


def projector(ms: xp.ndarray, left: bool, Ovlp_inv1: xp.ndarray =None, Ovlp0: xp.ndarray =None) -> xp.ndarray:
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
            proj = xp.tensordot(Ovlp_inv1, proj,  axes=(-1, 0))
            proj = xp.tensordot(proj, ms.conj(), axes=(0, 0))

    if left:
        sz = int(np.prod(ms.shape[:-1]))
    else:
        sz = int(np.prod(ms.shape[1:]))
    Iden = xp.array(xp.diag(xp.ones(sz)), dtype=backend.real_dtype).reshape(proj.shape)
    proj = Iden - proj
    return proj


# Note: don't do "optimization" like this. The contraction will take more time
"""
def hop_factory(ltensor, rtensor, mo, dim):
    h = opt_einsum.contract("abc, bdeg, fgh -> adfceh", ltensor, mo, rtensor)
    if dim == 3:
        # S-a   f-S
        #     d
        # O-b-O-g-O
        #     e
        # S-c   h-S
        def hop(ms):
            return np.tensordot(h, ms, 3)
    elif dim == 4:
        # S-a   f-S
        #     d
        # O-b-O-g-O
        #     e
        # S-c   h-S
        #     i
        def hop(ms):
            return np.tensordot(h, ms, [[3, 4, 5], [0, 1, 3]]).transpose([0, 1, 3, 2])
    else:
        assert False
    return hop
"""


def hop_factory(
    ltensor: Union[Matrix, xp.ndarray],
    rtensor: Union[Matrix, xp.ndarray],
    mo: Union[Matrix, xp.ndarray],
    ndim,
):
    if isinstance(ltensor, Matrix):
        ltensor = ltensor.array
    if isinstance(rtensor, Matrix):
        rtensor = rtensor.array
    if isinstance(mo, Matrix):
        mo = mo.array
    # S-a   l-S
    #     d
    # O-b-O-f-O
    #     e
    # S-c   k-S
    if ndim == 3:
        path = [
            ([0, 1], "abc, cek -> abek"),
            ([2, 0], "abek, bdef -> akdf"),
            ([1, 0], "akdf, lfk -> adl"),
        ]

        def hop(ms: xp.ndarray):
            return multi_tensor_contract(path, ltensor, ms, mo, rtensor)

        # S-a   l-S
        #     d
        # O-b-O-f-O
        #     e
        # S-c   k-S
        #     g
    elif ndim == 4:
        path = [
            ([0, 1], "abc, bdef -> acdef"),
            ([2, 0], "acdef, cegk -> adfgk"),
            ([1, 0], "adfgk, lfk -> adgl"),
        ]

        def hop(ms: xp.ndarray):
            return multi_tensor_contract(path, ltensor, mo, ms, rtensor)

    else:
        assert False

    return hop


def integrand_func_factory(shape, hop, islast, S_inv: xp.ndarray, left: bool,
        coef: complex, Ovlp_inv1: xp.ndarray =None, Ovlp_inv0: xp.ndarray =None, Ovlp0: xp.ndarray =None):
    # left == True: projector operate on the left side of the HC
    # Ovlp0 is (#.conj, #), Ovlp_inv0 = (#, #.conj), Ovlp_inv1 = (#, #.conj)
    # S_inv is (#.conj, #)
    def func(t, y):
        y0 = y.reshape(shape)
        HC = hop(y0)
        if not islast:
            proj = projector(y0, left, Ovlp_inv1, Ovlp0)
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
            if Ovlp_inv0 is not None:
                HC = tensordot(Ovlp_inv0, HC, axes=(-1, 0))
            return tensordot(HC, S_inv, axes=(-1, 0)).ravel() / coef
        else:
            if Ovlp_inv0 is not None:
                HC = tensordot(HC, Ovlp_inv0, axes=(-1, -1))
            return tensordot(S_inv, HC, axes=(0, 0)).ravel() / coef
        
    return func


def transferMat(mps, mpsconj, domain, imps, val):
    """
    calculate the transfer matrix from the left hand or the right hand
    """
    
    if mps[0].ndim == 3:
        if domain == "R":
            val = tensordot(mpsconj[imps], val, axes=(2, 0))
            val = tensordot(val, mps[imps], axes=([1, 2], [1, 2]))
        elif domain == "L":
            val = tensordot(mpsconj[imps], val, axes=(0, 0))
            val = tensordot(val, mps[imps], axes=([0, 2], [1, 0]))
        else:
            assert False
    
    elif mps[0].ndim == 4:
        if domain == "R":
            val = tensordot(mpsconj[imps], val, axes=(3, 0))
            val = tensordot(val, mps[imps], axes=([1, 2, 3], [1, 2, 3]))
        elif domain == "L":
            val = tensordot(mpsconj[imps], val, axes=(0, 0))
            val = tensordot(val, mps[imps], axes=([0, 3, 1], [1, 0, 2]))
        else:
            assert False
    else:
        raise ValueError(f"the dim of local mps is not correct: {mps[0].ndim}")

    return val


def _mu_regularize(s, epsilon=1e-10):
    """
    regularization of the singular value of the reduced density matrix
    """
    epsilon = np.sqrt(epsilon)
    return s + epsilon * np.exp(- s / epsilon)


class BraKetPair:
    def __init__(self, bra_mps, ket_mps, mpo=None):
        # do copy so that clear_memory won't clear previous braket
        self.bra_mps = bra_mps.copy()
        self.ket_mps = ket_mps.copy()
        self.mpo = mpo
        self.ft = self.calc_ft()

    def calc_ft(self):
        if self.mpo is None:
            dot = self.bra_mps.conj().dot(self.ket_mps)
        else:
            dot = self.bra_mps.conj().expectation(self.mpo, self.ket_mps)
        return (
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

    # todo: not used?
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
