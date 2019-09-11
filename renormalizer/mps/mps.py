# -*- encoding: utf-8 -*-

import logging
from functools import wraps
from typing import Union

import numpy as np
import scipy
from scipy import stats
from cached_property import cached_property


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

cached_property_set = set()


def _cached_property(func):
    cached_property_set.add(func.__name__)
    return cached_property(func)


def invalidate_cache_decorator(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        ret = f(self, *args, **kwargs)
        assert isinstance(ret, self.__class__)
        ret.invalidate_cache()
        return ret

    return wrapper


def adaptive_tdvp(fun):
    # evolve t/2 for 2 times
    #  J. Chem. Phys. 146, 174107 (2017)
    @wraps(fun)
    def f(self: "Mps", mpo, evolve_dt):
        if not self.evolve_config.adaptive:
            return fun(self, mpo, evolve_dt)
        config = self.evolve_config
        config.check_valid_dt(evolve_dt)
        accumulated_dt = 0
        # use 2 descriptors to decide accept or not: angle and energy
        # the logic about energy is different with that of prop&compress
        # because here we can compare energies early and restart early
        mps = None  # the mps after config.evolve_dt
        start = self  # the mps to start with
        start_energy = start.expectation(mpo)
        while True:
            logger.debug(f"adaptive dt: {config.evolve_dt}")
            mps_half1 = fun(start, mpo, config.evolve_dt / 2)._dmrg_normalize()
            e_half1 = mps_half1.expectation(mpo)
            if config.d_energy / 2 < abs(e_half1 - start_energy):
                # not converged
                logger.debug(
                    f"energy not converged in the first sub-step. start energy: {start_energy}, new energy: {e_half1}"
                )
                config.evolve_dt /= 2
                mps = mps_half1
                continue
            mps_half2 = fun(mps_half1, mpo, config.evolve_dt / 2)._dmrg_normalize()
            e_half2 = mps_half2.expectation(mpo)
            if config.d_energy < abs(e_half2 - start_energy):
                # not converged
                logger.debug(
                    f"energy not converged in the second sub-step. start energy: {start_energy}, new energy: {e_half2}"
                )
                config.evolve_dt /= 2
                mps = mps_half1
                continue
            if mps is None:
                mps = fun(start, mpo, config.evolve_dt)._dmrg_normalize()
            angle = mps.angle(mps_half2)
            logger.debug(
                f"Adaptive TDVP. angle: {angle}, start_energy: {start_energy}, e_half1: {e_half1}, e_half2: {e_half2}"
            )
            if 0.99995 < angle < 1.00005:
                # converged
                accumulated_dt += config.evolve_dt
                logger.debug(
                    f"evolution converged with dt: {config.evolve_dt}, accumulated: {accumulated_dt}"
                )
                if np.isclose(accumulated_dt, evolve_dt):
                    break
                start = mps_half2
                start_energy = e_half2
                mps = None
            else:
                # not converged
                config.evolve_dt /= 2
                logger.debug(f"evolution not converged, angle: {angle}")
                if abs(config.evolve_dt) / abs(evolve_dt - accumulated_dt) < 1e-2:
                    raise RuntimeError("too many sub-steps required in a single step")
                mps = mps_half1
        if 0.99999 < angle < 1.00001:
            # a larger dt could be used
            config.enlarge_evolve_dt()
            logger.debug(
                f"evolution easily converged, new evolve_dt: {config.evolve_dt}"
            )
            mps_half2.evolve_config = config
        return mps_half2

    return f

"""
Tried to just use substeps. Result not so good. Can't control step well.
Sometimes steps too large, sometimes too small
def adaptive_tdvp(fun):
    @functools.wraps(fun)
    def f(self: "Mps", mpo, evolve_dt):
        if not self.evolve_config.adaptive:
            return fun(self, mpo, evolve_dt)
        if evolve_dt < 0:
            raise NotImplementedError("adaptive tdvp with negative evolve dt not implemented")
        config = self.evolve_config
        # requires exactly divisible
        assert evolve_dt % config.evolve_dt == 0
        accumulated_dt = 0
        # Tried to use angle and energy as adaptive descriptor. Angle not working well, the
        # threshold is too flexible. A simple, useful threshold that balances accuracy and
        # time cost can't be found. Energy is too simple, can't guarantee accuracy.
        start = self  # the mps to start with
        while True:
            start_energy = start.expectation(mpo)
            logger.debug(f"adaptive dt: {config.evolve_dt}, start energy: {start_energy}")
            mps = fun(start, mpo, config.evolve_dt)
            energy = mps.expectation(mpo)
            stat: DescribeResult = mps.evolve_config.stat
            if 1e-3 < abs(energy - start_energy) or 4 < stat.mean:
                # not converged
                logger.debug(f"tdvp not converged, energy: {energy}")
                if config.evolve_dt / (evolve_dt - accumulated_dt) < 1e-2:
                    raise RuntimeError("too many sub-steps required in a single step")
                config.evolve_dt /= 2
                continue
            else:
                # converged
                accumulated_dt += config.evolve_dt
                logger.debug(
                    f"evolution converged with dt: {config.evolve_dt}, accumulated: {accumulated_dt}"
                )
                if np.isclose(accumulated_dt, evolve_dt):
                    break
                start = mps
        if mps.evolve_config.stat.mean < 3:
            # a larger dt could be used
            config.enlarge_evolve_dt()
            logger.debug(
                f"evolution easily converged, new evolve_dt: {config.evolve_dt}"
            )
            mps.evolve_config = config
        return mps

    return f
"""

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
        mps.left = False
        if mol_list.scheme < 4:
            mps.qntot = nexciton
        elif mol_list.scheme == 4:
            mps.qntot = 0
        else:
            assert False

        # print("self.dim", self.dim)

        mps.wfns = []
        for mol in mps.mol_list:
            for ph in mol.hartree_phs:
                mps.wfns.append(np.random.random(ph.n_phys_dim))
        mps.wfns.append(1.0)

        return mps

    @classmethod
    def gs(cls, mol_list: MolList, max_entangled: bool):
        """
        T = \\infty maximum entangled GS state
        electronic site: pbond 0 element 1.0
                         pbond 1 element 0.0
        phonon site: digonal element sqrt(pbond) for normalization
        """
        mps = cls()
        mps.mol_list = mol_list
        mps.qn = [[0]] * (len(mps.ephtable) + 1)
        mps.qnidx = len(mps.ephtable) - 1
        mps.left = False
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
                    mps.append(np.zeros((1, mol_list.mol_num, 1)))
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

        mps.wfns = []

        for mol in mol_list:
            for ph in mol.hartree_phs:
                if max_entangled:
                    diag_elems = [1.0] * ph.n_phys_dim
                    mps.wfns.append(np.diag(diag_elems))
                else:
                    diag_elems = [1.0] + [0.0] * (ph.n_phys_dim - 1)
                    mps.wfns.append(np.array(diag_elems))
        # the coefficent a
        mps.wfns.append(1.0)

        mflib.normalize(mps.wfns, 1.0)

        return mps

    def __init__(self):
        super(Mps, self).__init__()
        # todo: tdh part with GPU backend
        self.wfns = [1]

        self.optimize_config: OptimizeConfig = OptimizeConfig()
        self.evolve_config: EvolveConfig = EvolveConfig()

    def conj(self):
        new_mps = super().conj()
        for idx, wfn in enumerate(new_mps.wfns):
            new_mps.wfns[idx] = np.conj(wfn)
        return new_mps

    def dot(self, other, with_hartree=True):
        e = super(Mps, self).dot(other)
        if with_hartree:
            assert len(self.wfns) == len(other.wfns)
            for wfn1, wfn2 in zip(self.wfns[:-1], other.wfns[:-1]):
                # using vdot is buggy here, because vdot will take conjugation automatically
                e *= np.dot(wfn1, wfn2)
        return e

    def to_complex(self, inplace=False):
        new_mp = super(Mps, self).to_complex(inplace=inplace)
        new_mp.wfns = [wfn.astype(np.complex128) for wfn in new_mp.wfns[:-1]] + [
            new_mp.wfns[-1]
        ]
        return new_mp

    def _get_sigmaqn(self, idx):
        if self.ephtable.is_electron(idx):
            return [0, 1]
        elif self.ephtable.is_phonon(idx):
            return [0] * self.pbond_list[idx]
        else:
            if self.mol_list.scheme == 4:
                return [0] * self.pbond_list[idx]
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
        return self.wfns[-1]

    @property
    def hybrid_tdh(self):
        return not self.mol_list.pure_dmrg

    @property
    def nexciton(self):
        return self.qntot

    @property
    def norm(self):
        # return self.dmrg_norm * self.hartree_norm
        return self.wfns[-1]

    # @_cached_property
    @property
    def dmrg_norm(self):
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
        replacement_idx = self.mol_list.e_idx()
        orig_ms = self[replacement_idx]
        if self.mol_list.scheme == 4:
            ms = orig_ms.copy()
            if xp.linalg.norm(ms.array) == 0:
                assert ms.shape[0] == ms.shape[-1] == 1
                if self.is_mps:
                    ms[0, 0, 0] = 1
                elif self.is_mpdm:
                    ms[0, 0, 0, 0] = 1
                else:
                    assert False
                self[replacement_idx] = ms
        res = np.sqrt(self.conj().dot(self, with_hartree=False).real)
        assert res != 0
        self[replacement_idx] = orig_ms
        assert not np.iscomplex(res)
        return res.real

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

    def expectation(self, mpo, self_conj=None) -> float:
        if self_conj is None:
            self_conj = self._expectation_conj()
        environ = Environ(self, mpo, "R", mps_conj=self_conj)
        l = ones((1, 1, 1))
        r = environ.read("R", 1)
        path = self._expectation_path()
        return float(multi_tensor_contract(path, l, self[0], mpo[0], self_conj[0], r).real)
        # This is time and memory consuming
        # return self_conj.dot(mpo.apply(self), with_hartree=False).real

    def expectations(self, mpos) -> np.ndarray:
        if len(mpos) < 3:
            return np.array([self.expectation(mpo) for mpo in mpos])
        assert 2 < len(mpos)
        # id can be used as efficient hash because of `Matrix` implementation
        mpo_ids = np.array([[id(m) for m in mpo] for mpo in mpos])
        common_mpo_ids = mpo_ids[0].copy()
        mpo0_unique_idx = np.where(np.sum(mpo_ids == common_mpo_ids, axis=0) == 1)[0][0]
        common_mpo_ids[mpo0_unique_idx] = mpo_ids[1][mpo0_unique_idx]
        x, unique_idx = np.where(mpo_ids != common_mpo_ids)
        # should find one at each line
        assert np.allclose(x, np.arange(len(mpos)))
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

    @_cached_property
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

    @_cached_property
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

    @_cached_property
    def r_square(self):
        r_list = np.arange(0, self.mol_num)
        if np.allclose(self.e_occupations, np.zeros_like(self.e_occupations)):
            return 0
        r_mean_square = np.average(r_list, weights=self.e_occupations) ** 2
        mean_r_square = np.average(r_list ** 2, weights=self.e_occupations)
        return mean_r_square - r_mean_square

    def invalidate_cache(self):
        for p in cached_property_set:
            if p in self.__dict__:
                del self.__dict__[p]

    def metacopy(self) -> "Mps":
        new = super().metacopy()
        new.wfns = [wfn.copy() for wfn in self.wfns[:-1]] + [self.wfns[-1]]
        new.optimize_config = self.optimize_config
        # evolve_config has its own data
        new.evolve_config = self.evolve_config.copy()
        return new

    def calc_energy(self, h_mpo):
        return self.expectation(h_mpo)

    def clear_memory(self):
        # make a cache
        for prop in cached_property_set:
            _ = getattr(self, prop)
        self.clear()

    def _dmrg_normalize(self):
        return self.scale(1.0 / self.dmrg_norm, inplace=True)

    @invalidate_cache_decorator
    def normalize(self, norm=None):
        # real time propagation: dmrg should be normalized, tdh should be normalized, coefficient is not changed,
        #  use norm=None
        # imag time propagation: dmrg should be normalized, tdh should be normalized, coefficient is normalized to 1.0
        # applied by a operator then normalize: dmrg should be normalized,
        #   tdh should be normalized, coefficient is set to the length
        # these two cases should set `norm` equals to corresponding value
        self._dmrg_normalize()
        if norm is None:
            mflib.normalize(self.wfns, self.wfns[-1])
        else:
            mflib.normalize(self.wfns, norm)
        return self

    @invalidate_cache_decorator
    def canonical_normalize(self):
        # applied by a operator then normalize: dmrg should be normalized,
        #   tdh should be normalized, coefficient is set to the length
        # suppose length is only determined by dmrg part
        return self.normalize(self.dmrg_norm)

    def evolve(self, mpo, evolve_dt, approx_eiht=None):
        if self.hybrid_tdh:
            hybrid_mpo, HAM, Etot = self.construct_hybrid_Ham(mpo)
            mps = self.evolve_dmrg(hybrid_mpo, evolve_dt, approx_eiht)
            unitary_propagation(mps.wfns, HAM, Etot, evolve_dt)
        else:
            # save the cost of calculating energy
            mps = self.evolve_dmrg(mpo, evolve_dt, approx_eiht)
        if np.iscomplex(evolve_dt):
            mps.normalize(1.0)
        else:
            mps.normalize(None)
        return mps

    def evolve_dmrg(self, mpo, evolve_dt, approx_eiht=None) -> "Mps":
        if approx_eiht is not None:
            return approx_eiht.contract(self)

        if self.evolve_config.method == EvolveMethod.prop_and_compress:
            new_mps = self._evolve_dmrg_prop_and_compress(mpo, evolve_dt)
            return new_mps

        if self.evolve_config.should_adjust_bond_dim:
            logger.debug("adjusting bond order")
            # use this custom compress method for tdvp
            orig_compress_config: CompressConfig = self.compress_config.copy()
            self.compress_config.criteria = CompressCriteria.fixed
            self.compress_config.set_bonddim(len(self) + 1)
            config = self.evolve_config.copy()
            config.adaptive = True
            config.evolve_dt = evolve_dt
            self.compress_add = True
            new_mps = self._evolve_dmrg_prop_and_compress(mpo, evolve_dt, config)
            self.compress_config = new_mps.compress_config = orig_compress_config
            # won't be using that much memory anymore after the P&C
            backend.free_all_blocks()
            return new_mps

        method_mapping = {
            EvolveMethod.tdvp_mctdh: self._evolve_dmrg_tdvp_mctdh,
            EvolveMethod.tdvp_mctdh_new: self._evolve_dmrg_tdvp_mctdhnew,
            EvolveMethod.tdvp_ps: self._evolve_dmrg_tdvp_ps,
            EvolveMethod.tdvp_mu: self._evolve_dmrg_tdvp_mu,
        }
        method = method_mapping[self.evolve_config.method]
        new_mps = method(mpo, evolve_dt)
        return new_mps

    def _evolve_dmrg_prop_and_compress(self, mpo, evolve_dt, config: EvolveConfig = None) -> "Mps":
        if config is None:
            config = self.evolve_config
        assert evolve_dt is not None
        propagation_c = config.rk_config.coeff
        termlist = [self]
        # don't let bond dim grow when contracting
        orig_compress_config = self.compress_config
        contract_compress_config = self.compress_config.copy()
        if contract_compress_config.criteria is CompressCriteria.threshold:
            contract_compress_config.criteria = CompressCriteria.both
        contract_compress_config.min_dims = None
        contract_compress_config.max_dims = np.array(self.bond_dims) + 4
        self.compress_config = contract_compress_config
        while len(termlist) < len(propagation_c):
            termlist.append(mpo.contract(termlist[-1]))
        # bond dim can grow after adding
        for t in termlist:
            t.compress_config = orig_compress_config
        if config.adaptive:
            config.check_valid_dt(evolve_dt)
            while True:
                scaled_termlist = []
                for idx, term in enumerate(termlist):
                    scale = (-1.0j * config.evolve_dt) ** idx * propagation_c[idx]
                    scaled_termlist.append(term.scale(scale))
                del term
                new_mps1 = compressed_sum(scaled_termlist[:-1])._dmrg_normalize()
                new_mps2 = compressed_sum([new_mps1, scaled_termlist[-1]])._dmrg_normalize()
                angle = new_mps1.angle(new_mps2)
                energy1 = self.expectation(mpo)
                energy2 = new_mps1.expectation(mpo)
                rtol = config.adaptive_rtol  # default to 1e-3
                p = (rtol / (np.sqrt(2 * abs(1 - angle)) + 1e-30)) ** 0.2 * 0.8
                logger.debug(f"angle: {angle}. e1: {energy1}. e2: {energy2}, p: {p}")
                d_energy = config.d_energy
                if abs(energy1 - energy2) < d_energy and 0.5 < p:
                    # converged
                    if abs(config.evolve_dt - evolve_dt) / abs(evolve_dt) < 1e-5:
                        # equal evolve_dt
                        if abs(energy1 - energy2) < (d_energy/10) and 1.1 < p:
                            # a larger dt could be used
                            config.evolve_dt *= min(p, 1.5)
                            logger.debug(
                                f"evolution easily converged, new evolve_dt: {config.evolve_dt}"
                            )
                        # First exit
                        new_mps2.evolve_config.evolve_dt = config.evolve_dt
                        return new_mps2
                    if abs(config.evolve_dt) < abs(evolve_dt):
                        # step smaller than required
                        new_dt = evolve_dt - config.evolve_dt
                        logger.debug(f"remaining: {new_dt}")
                        # Second exit
                        new_mps2.evolve_config.evolve_dt = config.evolve_dt
                        del new_mps1, termlist, scaled_termlist  # memory consuming and not useful anymore
                        return new_mps2._evolve_dmrg_prop_and_compress(mpo, new_dt, config)
                    else:
                        # shouldn't happen.
                        raise ValueError(
                            f"evolve_dt in config: {config.evolve_dt}, in arg: {evolve_dt}"
                        )
                else:
                    # not converged
                    config.evolve_dt /= 2
                    logger.debug(
                        f"evolution not converged, new evolve_dt: {config.evolve_dt}"
                    )
        else:
            for idx, term in enumerate(termlist):
                term.scale(
                    (-1.0j * evolve_dt) ** idx * propagation_c[idx], inplace=True
                )
            return compressed_sum(termlist)

    def _evolve_dmrg_tdvp_mctdh(self, mpo, evolve_dt) -> "Mps":
        # TDVP for original MCTDH
        if self.is_right_canon:
            assert self.check_right_canonical()
            self.canonicalise()

        # a workaround for https://github.com/scipy/scipy/issues/10164
        imag_time = np.iscomplex(evolve_dt)
        if imag_time:
            evolve_dt = -evolve_dt.imag
            # used in calculating derivatives
            coef = -1
        else:
            coef = 1j

        # qn for this method has not been implemented
        self.use_dummy_qn = True
        self.clear_qn()
        mps = self.to_complex(inplace=True)
        mps_conj = mps.conj()
        environ = Environ(mps, mpo, "R")

        # initial matrix
        ltensor = np.ones((1, 1, 1))
        rtensor = np.ones((1, 1, 1))

        new_mps = self.metacopy()

        cmf_rk_steps = []

        for imps in range(len(mps)):
            ltensor = environ.GetLR(
                "L", imps - 1, mps, mpo, itensor=ltensor, method="System"
            )
            rtensor = environ.GetLR(
                "R", imps + 1, mps, mpo, itensor=rtensor, method="Enviro"
            )
            # density matrix
            S = transferMat(mps, mps_conj, "R", imps + 1).asnumpy()

            epsilon = 1e-8
            w, u = scipy.linalg.eigh(S)
            try:
                w = w + epsilon * np.exp(-w / epsilon)
            except FloatingPointError:
                logger.warning(f"eigenvalue of density matrix contains negative value")
                w -= 2 * w.min()
                w = w + epsilon * np.exp(-w / epsilon)
            # print
            # "sum w=", np.sum(w)
            # S  = u.dot(np.diag(w)).dot(np.conj(u.T))
            S_inv = xp.asarray(u.dot(np.diag(1.0 / w)).dot(np.conj(u.T)))

            # pseudo inverse
            # S_inv = scipy.linalg.pinvh(S,rcond=1e-2)

            shape = mps[imps].shape

            hop = hop_factory(ltensor, rtensor, mpo[imps], len(shape))

            func = integrand_func_factory(shape, hop, imps == len(mps) - 1, S_inv, True, coef)

            sol = solve_ivp(
                func, (0, evolve_dt), mps[imps].ravel().array, method="RK45"
            )
            # print
            # "CMF steps:", len(sol.t)
            cmf_rk_steps.append(len(sol.t))
            new_mps[imps] = sol.y[:, -1].reshape(shape)
            new_mps[imps].check_lortho()
            # print
            # "orthogonal1", np.allclose(np.tensordot(MPSnew[imps],
            #                                        np.conj(MPSnew[imps]), axes=([0, 1], [0, 1])),
            #                           np.diag(np.ones(MPSnew[imps].shape[2])))
        steps_stat = stats.describe(cmf_rk_steps)
        logger.debug(f"TDVP-MCTDH CMF steps: {steps_stat}")

        return new_mps

    @adaptive_tdvp
    def _evolve_dmrg_tdvp_mctdhnew(self, mpo, evolve_dt) -> "Mps":
        # new regularization scheme
        # JCP 148, 124105 (2018)
        # JCP 149, 044119 (2018)

        # a workaround for https://github.com/scipy/scipy/issues/10164
        imag_time = np.iscomplex(evolve_dt)
        if imag_time:
            evolve_dt = -evolve_dt.imag
            # used in calculating derivatives
            coef = -1
        else:
            coef = 1j

        imag_time = np.iscomplex(evolve_dt)


        # `self` should not be modified during the evolution
        # mps: the mps at time 0
        # environ_mps: mps to construct environ
        if imag_time:
            mps = self.copy()
        else:
            mps = self.to_complex()
        if self.evolve_config.tdvp_mctdh_cmf:
            # constant environment
            environ_mps = mps
            mps_t = mps.metacopy()
            sweep_round = 1
        else:
            # evolving environment
            environ_mps = mps
            mps_t = mps
            sweep_round = 2
        # construct the environment matrix
        environ = Environ(environ_mps, mpo)


        # statistics for debug output
        cmf_rk_steps = []

        for i in range(sweep_round):
            for imps in mps.iter_idx_list(full=True):
                shape = list(mps[imps].shape)

                system = "L" if mps.left else "R"

                qnbigl, qnbigr = mps._get_big_qn(imps)
                u, s, qnlset, v, s, qnrset = svd_qn.Csvd(
                    mps[imps].asnumpy(),
                    qnbigl,
                    qnbigr,
                    mps.qntot,
                    system=system,
                    full_matrices=False,
                )
                vt = v.T
                regular_s = _mu_regularize(s)

                if not mps.left:
                    islast = imps == 0
                    mps[imps] = vt.reshape([-1] + shape[1:])

                    ltensor = environ.read("L", imps - 1)
                    rtensor = environ.GetLR(
                        "R", imps + 1, environ_mps, mpo, itensor=None, method="System"
                    )

                    us = Matrix(u.dot(np.diag(s)))

                    ltensor = tensordot(ltensor, us, axes=(2, 0))
                    ltensor = tensordot(Matrix(u.conj()), ltensor, axes=(0, 0))

                    if not islast:
                        mps[imps - 1] = tensordot(mps[imps - 1], us , axes=(-1, 0))
                        mps.qn[imps] = qnrset
                        mps_t.qn[imps] = qnrset.copy()

                elif mps.left:
                    islast = imps == len(mps) - 1
                    mps[imps] = u.reshape(shape[:-1] + [-1])

                    ltensor = environ.GetLR(
                        "L", imps - 1, environ_mps, mpo, itensor=None, method="System"
                    )
                    rtensor = environ.read("R", imps + 1)

                    svt = Matrix(np.diag(s).dot(vt))

                    rtensor = tensordot(rtensor, svt, axes=(2, 1))
                    rtensor = tensordot(Matrix(vt.conj()), rtensor, axes=(1, 0))

                    if not islast:
                        mps[imps + 1] = tensordot(svt, mps[imps + 1], axes=(-1, 0))
                        mps.qn[imps + 1] = qnlset
                        mps_t.qn[imps + 1] = qnlset.copy()
                else:
                    assert False

                S_inv = xp.diag(1.0 / regular_s)

                hop = hop_factory(ltensor, rtensor, mpo[imps], len(shape))

                func = integrand_func_factory(shape, hop, islast, S_inv, mps.left, coef)

                sol = solve_ivp(
                    func, (0, evolve_dt / sweep_round), mps[imps].ravel().array, method="RK45"
                )
                cmf_rk_steps.append(len(sol.t))
                ms = sol.y[:, -1].reshape(shape)
                if islast:
                    if not mps.left:
                        ms = xp.tensordot(us.array, ms, 1)
                    else:
                        ms = xp.tensordot(ms, svt.array, 1)
                mps_t[imps] = ms
            if self.evolve_config.tdvp_mctdh_cmf:
                # environ_mps == mps, mps_t = mps.copy()
                mps._switch_direction()
                mps_t._switch_direction()
            else:
                # environ_mps == mps == mps_t
                mps_t._switch_direction()
        steps_stat = stats.describe(cmf_rk_steps)
        logger.debug(f"TDVP-MCTDH CMF steps: {steps_stat}")
        # new_mps.evolve_config.stat = steps_stat

        return mps_t


    @adaptive_tdvp
    def _evolve_dmrg_tdvp_mu(self, mpo, evolve_dt) -> "Mps":
        # new regularization scheme
        # JCP 148, 124105 (2018)
        # JCP 149, 044119 (2018)

        # a workaround for https://github.com/scipy/scipy/issues/10164
        imag_time = np.iscomplex(evolve_dt)
        if imag_time:
            evolve_dt = -evolve_dt.imag
            # used in calculating derivatives
            coef = -1
        else:
            coef = 1j

        imag_time = np.iscomplex(evolve_dt)

        if self.is_right_canon:
            assert self.check_right_canonical()
            self.canonicalise()
        else:
            if not self.check_left_canonical():
                self.move_qnidx(0)
                self.left = True
                self.canonicalise()
                assert self.check_left_canonical()

        # `self` should not be modified during the evolution
        # mps: the mps to return
        # environ_mps: mps to construct environ
        if imag_time:
            mps = self.copy()
        else:
            mps = self.to_complex()


        if self.evolve_config.tdvp_mu_midpoint:
            # mps at t/2 as environment
            orig_config = self.evolve_config.copy()
            self.evolve_config.tdvp_mu_midpoint = False
            environ_mps = self.evolve(mpo, evolve_dt / 2)
            self.evolve_config = orig_config
        else:
            # mps at t=0 as environment
            environ_mps = mps.copy()
        # construct the environment matrix
        environ = Environ(environ_mps, mpo, "L")
        environ.write_r_sentinel(environ_mps)

        # statistics for debug output
        cmf_rk_steps = []

        for imps in mps.iter_idx_list(full=True):
            shape = list(mps[imps].shape)
            ltensor = environ.read("L", imps - 1)
            if imps == self.site_num - 1:
                # the coefficient site
                rtensor = ones((1, 1, 1))
                hop = hop_factory(ltensor, rtensor, mpo[imps], len(shape))

                def func(y):
                    return hop(y.reshape(shape)).ravel()

                ms = expm_krylov(func, evolve_dt / coef, mps[imps].ravel().array)
                mps[imps] = ms.reshape(shape)
                continue

            # perform qr on the environment mps
            qnbigl, qnbigr = environ_mps._get_big_qn(imps + 1)
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

            ltensor = environ.read("L", imps - 1)
            rtensor = environ.GetLR(
                "R", imps + 1, environ_mps, mpo, itensor=None, method="System"
            )

            regular_s = _mu_regularize(s)

            us = Matrix(u.dot(np.diag(s)))

            rtensor = tensordot(rtensor, us, axes=(-1, -1))

            environ_mps[imps] = tensordot(environ_mps[imps], us, axes=(-1, 0))
            environ_mps.qn[imps + 1] = qnrset

            S_inv = Matrix(u).conj().dot(xp.diag(1.0 / regular_s)).T

            hop = hop_factory(ltensor, rtensor, mpo[imps], len(shape))

            func = integrand_func_factory(shape, hop, False, S_inv.array, True, coef)

            sol = solve_ivp(
                func, (0, evolve_dt), mps[imps].ravel().array, method="RK45"
            )
            cmf_rk_steps.append(len(sol.t))
            ms = sol.y[:, -1].reshape(shape)
            mps[imps] = ms
        steps_stat = stats.describe(cmf_rk_steps)
        logger.debug(f"TDVP-MCTDH CMF steps: {steps_stat}")
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

        # statistics for debug output
        cmf_rk_steps = []
        USE_RK = self.evolve_config.tdvp_ps_rk4
        # sweep for 2 rounds
        for i in range(2):
            for imps in mps.iter_idx_list(full=True):
                system = "L" if mps.left else "R"
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
                    cmf_rk_steps.append(len(sol.t))
                    mps_t = sol.y[:, -1]
                else:
                    # Can't use the same func because here H should be Hermitian
                    def func(y):
                        return hop(y.reshape(shape)).ravel()
                    mps_t = expm_krylov(func, (evolve_dt / 2) / coef, mps[imps].ravel().array)
                mps_t = mps_t.reshape(shape)

                qnbigl, qnbigr = mps._get_big_qn(imps)
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

                if not mps.left and imps != 0:
                    mps[imps] = vt.reshape([-1] + shape[1:])
                    mps_conj[imps] = mps[imps].conj()
                    mps.qn[imps] = qnrset

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
                        cmf_rk_steps.append(len(sol_u.t))
                        mps_t = sol_u.y[:, -1]
                    else:
                        def func_u(y):
                            return hop_svt(y.reshape(shape_u)).ravel()
                        mps_t = expm_krylov(func_u, (-evolve_dt / 2) / coef, u.ravel())
                    mps_t = mps_t.reshape(shape_u)
                    mps[imps - 1] = tensordot(
                        mps[imps - 1].array,
                        mps_t,
                        axes=(-1, 0),
                    )
                    mps_conj[imps - 1] = mps[imps - 1].conj()

                elif mps.left and imps != len(mps) - 1:
                    mps[imps] = u.reshape(shape[:-1] + [-1])
                    mps_conj[imps] = mps[imps].conj()
                    mps.qn[imps + 1] = qnlset

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
                        cmf_rk_steps.append(len(sol_svt.t))
                        mps_t = sol_svt.y[:, -1]
                    else:
                        def func_svt(y):
                            return hop_svt(y.reshape(shape_svt)).ravel()
                        mps_t = expm_krylov(func_svt, (-evolve_dt / 2) / coef, vt.ravel())
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

        if USE_RK:
            steps_stat = stats.describe(cmf_rk_steps)
            logger.debug(f"TDVP-PS CMF steps: {steps_stat}")
            mps.evolve_config.stat = steps_stat

        return mps


    def evolve_exact(self, h_mpo, evolve_dt, space):
        MPOprop, HAM, Etot = self.hybrid_exact_propagator(
            h_mpo, -1.0j * evolve_dt, space
        )
        new_mps = MPOprop.apply(self, canonicalise=True)
        unitary_propagation(new_mps.wfns, HAM, Etot, evolve_dt)
        return new_mps

    @property
    def digest(self):
        if 10 < self.site_num:
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
        WFN = self.wfns
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
                e_mean = mflib.exp_value(self.wfns[iwfn], h_vib_indep, self.wfns[iwfn])
                if space == "EX":
                    e_mean += mflib.exp_value(
                        self.wfns[iwfn], h_vib_dep, self.wfns[iwfn]
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

    def hartree_wfn_diff(self, other):
        assert len(self.wfns) == len(other.wfns)
        res = []
        for wfn1, wfn2 in zip(self.wfns, other.wfns):
            res.append(
                scipy.linalg.norm(
                    np.tensordot(wfn1, wfn1, axes=0) - np.tensordot(wfn2, wfn2, axes=0)
                )
            )
        return np.array(res)

    def full_wfn(self):
        dim = np.prod(self.pbond_list)
        if 20000 < dim:
            raise ValueError("wavefunction too large")
        res = ones((1, 1, 1))
        for mt in self:
            dim1 = res.shape[1] * mt.shape[1]
            dim2 = mt.shape[-1]
            res = tensordot(res, mt, axes=1).reshape(1, dim1, dim2)
        return res[0, :, 0]

    def _calc_reduced_density_matrix(self, mp1, mp2):
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
        return reduced_density_matrix

    def calc_reduced_density_matrix(self) -> np.ndarray:
        if self.mol_list.scheme < 4:
            mp1 = [mt.reshape(mt.shape[0], mt.shape[1], 1, mt.shape[2]) for mt in self]
            mp2 = [mt.reshape(mt.shape[0], 1, mt.shape[1], mt.shape[2]).conj() for mt in self]
            return self._calc_reduced_density_matrix(mp1, mp2)
        elif self.mol_list.scheme == 4:
            # be careful this method should be read-only
            copy = self.copy()
            copy.canonicalise(self.mol_list.e_idx())
            e_mo = copy[self.mol_list.e_idx()]
            return tensordot(e_mo.conj(), e_mo, axes=((0, 2), (0, 2))).asnumpy()
        else:
            assert False


    def __str__(self):
        # too many digits in the default format
        e_occupations_str = ", ".join(
            ["%.2f" % number for number in self.e_occupations]
        )
        template_str = "current size: {}, Matrix product bond dim:{}, electron occupations: {}"
        return template_str.format(
            sizeof_fmt(self.total_bytes),
            self.bond_dims,
            e_occupations_str,
        )

    def __setitem__(self, key, value):
        self.invalidate_cache()
        return super().__setitem__(key, value)

    def __add__(self, other: "Mps"):
        return self.add(other)

    def __sub__(self, other: "Mps"):
        return self.add(other.scale(-1))


def projector(ms: xp.ndarray, left: bool) -> xp.ndarray:
    if left:
        axes = (-1, -1)
    else:
        axes = (0, 0)
    proj = xp.tensordot(ms, ms.conj(), axes=axes)
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


def integrand_func_factory(shape, hop, islast, S_inv: xp.ndarray, left: bool, coef: complex):
    # left == True: projector operate on the left side of the HC
    def func(t, y):
        y0 = y.reshape(shape)
        HC = hop(y0)
        if not islast:
            proj = projector(y0, left)
            if y0.ndim == 3:
                if left:
                    HC = tensordot(proj, HC, axes=([2, 3], [0, 1]))
                    # uncomment this might resolve some numerical problem
                    # HC = tensordot(proj, HC, axes=([2, 3], [0, 1]))
                else:
                    HC = tensordot(HC, proj, axes=([1, 2], [2, 3]))
            elif y0.ndim == 4:
                if left:
                    HC = tensordot(proj, HC, axes=([3, 4, 5], [0, 1, 2]))
                    # HC = tensordot(proj, HC, axes=([3, 4, 5], [0, 1, 2]))
                else:
                    HC = tensordot(HC, proj, axes=([1, 2, 3], [3, 4, 5]))

        if left:
            return tensordot(HC, S_inv, axes=(-1, 0)).ravel() / coef
        else:
            return tensordot(S_inv, HC, axes=(0, 0)).ravel() / coef

    return func


def transferMat(mps, mpsconj, domain, siteidx):
    """
    calculate the transfer matrix from the left hand or the right hand
    """
    val = ones([1, 1])
    if domain == "R":
        for imps in range(len(mps) - 1, siteidx - 1, -1):
            val = tensordot(mpsconj[imps], val, axes=(2, 0))
            val = tensordot(val, mps[imps], axes=([1, 2], [1, 2]))
    elif domain == "L":
        for imps in range(0, siteidx + 1, 1):
            val = tensordot(mpsconj[imps], val, axes=(0, 0))
            val = tensordot(val, mps[imps], axes=([0, 2], [1, 0]))

    return val


def _mu_regularize(s):
    epsilon = 1e-10
    epsilon = np.sqrt(epsilon)
    return s + epsilon * np.exp(- s / epsilon)


class BraKetPair:
    def __init__(self, bra_mps, ket_mps, mpo=None):
        # do copy so that clear_memory won't clear previous braket
        self.bra_mps = bra_mps.copy()
        self.ket_mps = ket_mps.copy()
        self.mpo = mpo
        # for adaptive evolution. This is not an ideal solution but
        # I can't find anyone better. Bra and Ket have the same step size during
        # the evolution. Is this necessary?
        self.evolve_config = ket_mps.evolve_config
        self.ft = self.calc_ft()

    def calc_ft(self):
        if self.mpo is None:
            dot = self.bra_mps.conj().dot(self.ket_mps)
        else:
            dot = self.bra_mps.conj().dot(self.mpo.apply(self.ket_mps))
        return (
            dot * np.conjugate(self.bra_mps.coeff)
            * self.ket_mps.coeff
        )

    def clear_memory(self):
        self.bra_mps.clear_memory()
        self.ket_mps.clear_memory()

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