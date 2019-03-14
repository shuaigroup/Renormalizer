from __future__ import absolute_import, print_function, unicode_literals

import logging
import functools
from typing import List, Union, Tuple

import numpy as np
import scipy
from scipy import stats
from cached_property import cached_property


from ephMPS.lib import solve_ivp
from ephMPS.mps import svd_qn
from ephMPS.mps.matrix import (
    multi_tensor_contract,
    vstack,
    dstack,
    concatenate,
    zeros,
    ones,
    tensordot,
    Matrix,
    asnumpy,
)
from ephMPS.mps.backend import backend, xp
from ephMPS.mps.lib import Environ, updatemps, compressed_sum
from ephMPS.mps.mp import MatrixProduct
from ephMPS.mps.mpo import Mpo
from ephMPS.mps.tdh import mflib
from ephMPS.mps.tdh import unitary_propagation
from ephMPS.utils import (
    Quantity,
    OptimizeConfig,
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
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        ret = f(self, *args, **kwargs)
        assert isinstance(ret, self.__class__)
        ret.invalidate_cache()
        return ret

    return wrapper

def adaptive_tdvp(fun):
    @functools.wraps(fun)
    def f(self: "Mps", mpo, evolve_dt):
        if not self.evolve_config.adaptive:
            return fun(self, mpo, evolve_dt)
        config = self.evolve_config
        accumulated_dt = 0
        # use 2 descriptors to decide accept or not: angle and energy
        mps = None  # the mps after config.evolve_dt
        start = self  # the mps to start with
        start_energy = start.expectation(mpo)
        while True:
            logger.debug(f"adaptive dt: {config.evolve_dt}")
            mps_half1 = fun(start, mpo, config.evolve_dt / 2)
            e_half1 = mps_half1.expectation(mpo)
            if 5e-4 < abs(e_half1 - start_energy):
                # not converged
                logger.debug(f"energy not converged in the first sub-step. start energy: {start_energy}, new energy: {e_half1}")
                config.evolve_dt /= 2
                mps = mps_half1
                continue
            mps_half2 = fun(mps_half1, mpo, config.evolve_dt / 2)
            e_half2 = mps_half2.expectation(mpo)
            if 1e-3 < abs(e_half2 - start_energy):
                # not converged
                logger.debug(f"energy not converged in the second sub-step. start energy: {start_energy}, new energy: {e_half2}")
                config.evolve_dt /= 2
                mps = mps_half1
                continue
            if mps is None:
                mps = fun(start, mpo, config.evolve_dt)
            angle = mps.angle(mps_half2)
            logger.debug(f"Adaptive TDVP. angle: {angle}, start_energy: {start_energy}, e_half1: {e_half1}, e_half2: {e_half2}")
            if 0.999 < angle < 1.001:
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
                logger.debug(
                    f"evolution not converged, angle: {angle}"
                )
                if config.evolve_dt / (evolve_dt - accumulated_dt) < 1e-2:
                    raise RuntimeError("too many sub-steps required in a single step")
                mps = mps_half1
        if 0.99998 < angle < 1.00002:
            # a larger dt could be used
            config.evolve_dt *= 1.5
            logger.debug(
                f"evolution easily converged, new evolve_dt: {config.evolve_dt}"
            )
            mps_half2.evolve_config = config
        return mps_half2
    return f

class Mps(MatrixProduct):
    @classmethod
    def random(cls, mpo, nexciton, m_max, percent=0):
        mps = cls()
        mps.mol_list = mpo.mol_list
        mps.qn = [[0]]
        dim_list = [1]

        for imps in range(len(mpo) - 1):

            # quantum number
            qnbig = np.add.outer(mps.qn[imps], mps._get_sigmaqn(imps)).flatten()
            u_set = []
            s_set = []
            qnset = []

            for iblock in range(min(qnbig), nexciton + 1):
                # find the quantum number index
                indices = [i for i, x in enumerate(qnbig) if x == iblock]

                if len(indices) != 0:
                    a = np.random.random([len(indices), len(indices)]) - 0.5
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
                mt.reshape((dim_list[imps], mps.pbond_list[imps], dim_list[imps + 1]))
            )
            mps.qn.append(mpsqn)

        # the last site
        mps.qn.append([0])
        dim_list.append(1)
        last_mt = xp.random.random([dim_list[-2], mps.pbond_list[-1], dim_list[-1]]) - 0.5
        # normalize the mt so that the whole mps is normalized
        last_mt /= xp.linalg.norm(last_mt.flatten())
        mps.append(last_mt)

        mps.qnidx = len(mps) - 1
        mps.qntot = nexciton

        # print("self.dim", self.dim)

        mps.wfns = []
        for mol in mps.mol_list:
            for ph in mol.hartree_phs:
                mps.wfns.append(np.random.random(ph.n_phys_dim))
        mps.wfns.append(1.0)

        return mps

    @classmethod
    def gs(cls, mol_list, max_entangled):
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
        mps.qntot = 0

        for mol in mol_list:
            # electron mps
            mps.append(np.array([1, 0]).reshape(1, 2, 1))
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
        self.wfns = [1]

        self.optimize_config: OptimizeConfig = OptimizeConfig()
        self.evolve_config: EvolveConfig = EvolveConfig()

        self.compress_add: bool = False

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
        else:
            return [0] * self.pbond_list[idx]

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

    # todo: no need to cache this O(1) operation?
    @_cached_property
    def norm(self):
        # return self.dmrg_norm * self.hartree_norm
        return self.wfns[-1]

    # @_cached_property
    @property
    def dmrg_norm(self):
        # the fast version in the comment rarely makes sense because in a lot of cases
        # the mps is not canonicalised (though qnidx is set)
        """ Fast version yet not safe. Needs further testing
        if self.is_left_canon:
            assert self.check_left_canonical()
            return np.linalg.norm(np.ravel(self[-1]))
        else:
            assert self.check_right_canonical()
            return np.linalg.norm(np.ravel(self[0]))
        """
        return np.sqrt(self.conj().dot(self, with_hartree=False).real)

    def expectation(self, mpo, self_conj=None):
        # todo: different bra and ket
        if self_conj is None:
            self_conj = self.conj()
        environ = Environ()
        environ.construct(self, self_conj, mpo, "r")
        r = environ.read("r", 1)
        if self.is_mps:
            # g--S--h--S
            #    |     |
            #    e     |
            #    |     |
            # d--O--f--O
            #    |     |
            #    b     |
            #    |     |
            # a--S--c--S
            path = [
                ([0, 3], "abc, hfc -> abhf"),
                ([2, 0], "abhf, debf -> ahde"),
                ([1, 0], "ahde, geh -> adg"),
            ]
        elif self.is_mpdm:
            #    d
            #    |
            # h--S--j--S
            #    |     |
            #    f     |
            #    |     |
            # e--O--g--O
            #    |     |
            #    b     |
            #    |     |
            # a--S--c--S
            #    |
            #    d
            path = [
                ([0, 3], "abdc, jgc -> abdjg"),
                ([2, 0], "abdjg, efbg -> adjef"),
                ([1, 0], "adjef, hfdj -> aeh"),
            ]
        else:
            raise RuntimeError
        return float(multi_tensor_contract(path, self[0], mpo[0], self_conj[0], r).real)
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
        self_conj = self.conj()
        environ = Environ()
        environ.construct(self, self_conj, common_mpo, "l")
        environ.construct(self, self_conj, common_mpo, "r")
        res_list = []
        if self.is_mps:
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
        elif self.is_mpdm:
            #       e
            #       |
            # S--a--S--f--S
            # |     |     |
            # |     d     |
            # |     |     |
            # O--b--O--h--O
            # |     |     |
            # |     g     |
            # |     |     |
            # S--c--S--j--S
            #       |
            #       e
            path = [
                ([0, 1], "abc, cgej -> abgej"),
                ([3, 0], "abgej, bdgh -> aejdh"),
                ([2, 0], "aejdh, adef -> jhf"),
                ([1, 0], "jhf, fhj -> "),
            ]
        else:
            raise RuntimeError
        for idx, mpo in zip(unique_idx, mpos):
            l = environ.read("l", idx - 1)
            r = environ.read("r", idx + 1)
            res = multi_tensor_contract(path, l, self[idx], mpo[idx], self_conj[idx], r)
            res_list.append(float(res.real))
        return np.array(res_list)
        # the naive way
        # return np.array([self.expectation(mpo) for mpo in mpos])

    @_cached_property
    def ph_occupations(self):
        key = "ph_occupations"
        if key not in self.mol_list.mpos:
            mpos = []
            for imol, mol in enumerate(self.mol_list):
                for iph in range(len(mol.dmrg_phs)):
                    mpos.append(Mpo.ph_occupation_mpo(self.mol_list, imol, iph))
            self.mol_list.mpos[key] = mpos
        else:
            mpos = self.mol_list.mpos[key]
        return self.expectations(mpos)

    @_cached_property
    def e_occupations(self):
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

    @_cached_property
    def r_square(self):
        r_list = np.arange(0, self.mol_num)
        r_mean_square = np.average(r_list, weights=self.e_occupations) ** 2
        mean_r_square = np.average(r_list ** 2, weights=self.e_occupations)
        return mean_r_square - r_mean_square

    def invalidate_cache(self):
        for p in cached_property_set:
            if p in self.__dict__:
                del self.__dict__[p]

    def metacopy(self):
        new = super().metacopy()
        new.wfns = [wfn.copy() for wfn in self.wfns[:-1]] + [self.wfns[-1]]
        new.optimize_config = self.optimize_config
        new.evolve_config = self.evolve_config
        new.compress_add = self.compress_add
        return new

    def calc_energy(self, h_mpo):
        return self.expectation(h_mpo)

    def clear_memory(self):
        # make a cache
        for prop in cached_property_set:
            _ = getattr(self, prop)
        self.clear()

    def add(self, other):
        assert self.qntot == other.qntot
        assert self.site_num == other.site_num
        assert self.is_left_canon == other.is_left_canon

        new_mps = other.metacopy()
        new_mps.compress_config.update(self.compress_config)

        if self.is_mps:  # MPS
            new_mps[0] = dstack([self[0], other[0]])
            for i in range(1, self.site_num - 1):
                mta = self[i]
                mtb = other[i]
                pdim = mta.shape[1]
                assert pdim == mtb.shape[1]
                new_ms = zeros(
                    [mta.shape[0] + mtb.shape[0], pdim, mta.shape[2] + mtb.shape[2]],
                    dtype=mta.dtype,
                )
                new_ms[: mta.shape[0], :, : mta.shape[2]] = mta
                new_ms[mta.shape[0] :, :, mta.shape[2] :] = mtb
                new_mps[i] = new_ms

            new_mps[-1] = vstack([self[-1], other[-1]])
        elif self.is_mpdm:  # MPO
            new_mps[0] = concatenate((self[0], other[0]), axis=3)
            for i in range(1, self.site_num - 1):
                mta = self[i]
                mtb = other[i]
                pdimu = mta.shape[1]
                pdimd = mta.shape[2]
                assert pdimu == mtb.shape[1]
                assert pdimd == mtb.shape[2]

                new_ms = zeros(
                    [
                        mta.shape[0] + mtb.shape[0],
                        pdimu,
                        pdimd,
                        mta.shape[3] + mtb.shape[3],
                    ],
                    dtype=mta.dtype,
                )
                new_ms[: mta.shape[0], :, :, : mta.shape[3]] = mta[:, :, :, :]
                new_ms[mta.shape[0] :, :, :, mta.shape[3] :] = mtb[:, :, :, :]
                new_mps[i] = new_ms

            new_mps[-1] = concatenate((self[-1], other[-1]), axis=0)
        else:
            assert False

        new_mps.move_qnidx(self.qnidx)
        new_mps.qn = [qn1 + qn2 for qn1, qn2 in zip(self.qn, new_mps.qn)]
        new_mps.qn[0] = [0]
        new_mps.qn[-1] = [0]
        new_mps.set_peak_bytes()
        if self.compress_add:
            new_mps.canonicalise()
            new_mps.compress()
        return new_mps

    @invalidate_cache_decorator
    def normalize(self, norm=None):
        # real time propagation: dmrg should be normalized, tdh should be normalized, coefficient is not changed,
        #  use norm=None
        # imag time propagation: dmrg should be normalized, tdh should be normalized, coefficient is normalized to 1.0
        # applied by a operator then normalize: dmrg should be normalized,
        #   tdh should be normalized, coefficient is set to the length
        # these two cases should set `norm` equals to corresponding value
        self.scale(1.0 / self.dmrg_norm, inplace=True)
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

        if self.evolve_config.scheme == EvolveMethod.prop_and_compress:
            new_mps = self._evolve_dmrg_prop_and_compress(mpo, evolve_dt)
            if self.evolve_config.memory_limit < new_mps.peak_bytes:
                logger.info("switch to fixed bond order compression to save memory")
                new_mps.compress_config.set_runtime_bondorder(new_mps.bond_dims[1:-1])
            return new_mps

        if not self._tdvp_check_bond_order():
            orig_compress_config: CompressConfig = self.compress_config.copy()
            self.compress_config.set_bondorder(
                len(self) - 1, self.evolve_config.expected_bond_order
            )
            new_mps = self._evolve_dmrg_prop_and_compress(mpo, evolve_dt)
            self.compress_config = orig_compress_config
            return new_mps

        method_mapping = {
            EvolveMethod.tdvp_mctdh: self._evolve_dmrg_tdvp_mctdh,
            EvolveMethod.tdvp_mctdh_new: self._evolve_dmrg_tdvp_mctdhnew,
            EvolveMethod.tdvp_ps: self._evolve_dmrg_tdvp_ps}
        method = method_mapping[self.evolve_config.scheme]
        new_mps = method(mpo, evolve_dt)
        return new_mps

    def _evolve_dmrg_prop_and_compress(self, mpo, evolve_dt) -> "Mps":
        config = self.evolve_config
        propagation_c = self.evolve_config.rk_config.coeff
        termlist = [self]
        while len(termlist) < len(propagation_c):
            termlist.append(mpo.contract(termlist[-1]))
            # control term sizes to be approximately constant
            termlist[-1].compress_config.relax()
        if config.adaptive:
            while True:
                scaled_termlist = []
                for idx, term in enumerate(termlist):
                    scale = (-1.0j * config.evolve_dt) ** idx * propagation_c[idx]
                    scaled_termlist.append(term.scale(scale))
                del term
                new_mps1 = compressed_sum(scaled_termlist[:-1])
                new_mps2 = compressed_sum([new_mps1, scaled_termlist[-1]])
                angle = new_mps1.angle(new_mps2)
                logger.debug(f"angle: {angle:f}")
                # some tests show that five 9s mean totally safe
                # four 9s with last digit smaller than 5 mean unstably is coming
                # three 9s explode immediately
                if 0.99996 < angle < 1.00004:
                    # converged
                    if abs(config.evolve_dt - evolve_dt) / evolve_dt < 1e-5:
                        # equal evolve_dt
                        if 0.99999 < angle < 1.00001:
                            # a larger dt could be used
                            config.evolve_dt *= 1.5
                            logger.debug(
                                f"evolution easily converged, new evolve_dt: {config.evolve_dt}"
                            )
                        # First exit
                        new_mps2.evolve_config = config
                        return new_mps2
                    if config.evolve_dt < evolve_dt:
                        # step too small
                        new_dt = evolve_dt - config.evolve_dt
                        logger.debug(f"remaining: {new_dt}")
                        # Second exit
                        new_mps2.evolve_config = config
                        del new_mps1, termlist, scaled_termlist  # memory consuming and not used
                        return new_mps2._evolve_dmrg_prop_and_compress(mpo, new_dt)
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

    def _tdvp_check_bond_order(self) -> bool:
        assert self.evolve_config.scheme != EvolveMethod.prop_and_compress
        assert self.evolve_config.expected_bond_order is not None
        # neither too low nor too high will do
        return max(self.bond_dims) == self.evolve_config.expected_bond_order

    @adaptive_tdvp
    def _evolve_dmrg_tdvp_mctdh(self, mpo, evolve_dt) -> "Mps":
        # TDVP for original MCTDH
        if self.is_right_canon:
            assert self.check_right_canonical()
            self.canonicalise()
        # qn for this method has not been implemented
        self.use_dummy_qn = True
        self.clear_qn()
        mps = self.to_complex(inplace=True)
        mps_conj = mps.conj()
        environ = Environ()
        environ.construct(mps, mps_conj, mpo, "R")

        # initial matrix
        ltensor = np.ones((1, 1, 1))
        rtensor = np.ones((1, 1, 1))

        new_mps = self.metacopy()

        for imps in range(len(mps)):
            ltensor = environ.GetLR(
                "L", imps - 1, mps, mps_conj, mpo, itensor=ltensor, method="System"
            )
            rtensor = environ.GetLR(
                "R", imps + 1, mps, mps_conj, mpo, itensor=rtensor, method="Enviro"
            )
            # density matrix
            S = transferMat(mps, mps_conj, "R", imps + 1)

            epsilon = 1e-10
            w, u = scipy.linalg.eigh(S)
            w = w + epsilon * np.exp(-w / epsilon)
            # print
            # "sum w=", np.sum(w)
            # S  = u.dot(np.diag(w)).dot(np.conj(u.T))
            S_inv = xp.asarray(u.dot(np.diag(1.0 / w)).dot(np.conj(u.T)))

            # pseudo inverse
            # S_inv = scipy.linalg.pinvh(S,rcond=1e-2)

            shape = mps[imps].shape

            hop = hop_factory(ltensor, rtensor, mpo[imps], len(shape))

            func = integrand_func_factory(shape, hop, imps == len(mps) - 1, S_inv)

            sol = solve_ivp(
                func, (0, evolve_dt), mps[imps].ravel().array, method="RK45"
            )
            # print
            # "CMF steps:", len(sol.t)
            new_mps[imps] = sol.y[:, -1].reshape(shape)
            # print
            # "orthogonal1", np.allclose(np.tensordot(MPSnew[imps],
            #                                        np.conj(MPSnew[imps]), axes=([0, 1], [0, 1])),
            #                           np.diag(np.ones(MPSnew[imps].shape[2])))

        return new_mps

    @adaptive_tdvp
    def _evolve_dmrg_tdvp_mctdhnew(self, mpo, evolve_dt) -> "Mps":
        # new regularization scheme
        # JCP 148, 124105 (2018)
        # JCP 149, 044119 (2018)

        if self.is_left_canon:
            assert self.check_left_canonical()
            self.canonicalise()
        # qn for this method has not been implemented
        self.use_dummy_qn = True
        self.clear_qn()
        # xxx: uses 3x memory. Is it possible to only use 2x?
        mps = self.to_complex(inplace=True)

        # construct the environment matrix
        environ = Environ()
        environ.construct(mps, mps.conj(), mpo, "R")

        # initial matrix
        ltensor = ones((1, 1, 1))
        rtensor = ones((1, 1, 1))

        new_mps = mps.copy()

        for imps in range(len(mps)):
            shape = list(mps[imps].shape)

            u, s, vt = scipy.linalg.svd(
                mps[imps].reshape((-1, shape[-1])).asnumpy(), full_matrices=False
            )
            mps[imps] = u.reshape(shape[:-1] + [-1])

            ltensor = environ.GetLR(
                "L", imps - 1, mps, mps.conj(), mpo, itensor=ltensor, method="System"
            )
            rtensor = environ.GetLR(
                "R", imps + 1, mps, mps.conj(), mpo, itensor=rtensor, method="Enviro"
            )

            epsilon = 1e-10
            epsilon = np.sqrt(epsilon)
            s = s + epsilon * np.exp(-s / epsilon)

            svt = Matrix(np.diag(s).dot(vt))

            rtensor = tensordot(rtensor, svt, axes=(2, 1))
            rtensor = tensordot(Matrix(vt).conj(), rtensor, axes=(1, 0))

            if imps != len(mps) - 1:
                mps[imps + 1] = tensordot(svt, mps[imps + 1], axes=(-1, 0))

            # density matrix
            S = s * s
            # print
            # "sum density matrix", np.sum(S)

            S_inv = xp.diag(1.0 / s)

            hop = hop_factory(ltensor, rtensor, mpo[imps], len(shape))

            func = integrand_func_factory(shape, hop, imps == len(mps) - 1, S_inv)

            sol = solve_ivp(
                func, (0, evolve_dt), mps[imps].ravel().array, method="RK45"
            )
            # logger.debug(f"CMF steps: {len(sol.t)}")
            ms = sol.y[:, -1].reshape(shape)
            # check for othorgonal
            # e = np.tensordot(ms, ms, axes=((0, 1), (0, 1)))
            # print(np.allclose(np.eye(e.shape[0]), e))

            if imps == len(mps) - 1:
                # print
                # "s0", imps, s[0]
                new_mps[imps] = ms * s[0]
            else:
                new_mps[imps] = ms

                # print "orthogonal1", np.allclose(np.tensordot(MPSnew[imps],
                #    np.conj(MPSnew[imps]), axes=([0,1],[0,1])),
                #    np.diag(np.ones(MPSnew[imps].shape[2])))

        mps._switch_domain()
        new_mps._switch_domain()
        new_mps.canonicalise()

        return new_mps

    @adaptive_tdvp
    def _evolve_dmrg_tdvp_ps(self, mpo, evolve_dt) -> "Mps":
        # TDVP projector splitting
        mps = self.to_complex()  # make a copy
        # switch to make sure it's symmetric
        if mps.evolve_config.should_switch_side:
            mps.canonicalise()
        mps_conj = mps.conj()  # another copy, so 3x memory is used.

        # construct the environment matrix
        environ = Environ()
        # almost half is not used. Not a big deal.
        environ.construct(mps, mps_conj, mpo, "L")
        environ.construct(mps, mps_conj, mpo, "R")

        rk_steps = []
        # sweep for 2 rounds
        for i in range(2):
            for imps in mps.iter_idx_list(full=True):
                system = "R" if mps.is_left_canon else "L"
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

                def func(t, y):
                    return hop(y.reshape(shape)).ravel() / 1.0j

                sol = solve_ivp(
                    func, (0, evolve_dt / 2.0), mps[imps].ravel().array, method="RK45"
                )
                rk_steps.append(len(sol.t))
                mps_t = sol.y[:, -1].reshape(shape)
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

                if mps.is_left_canon and imps != 0:
                    mps[imps] = vt.reshape([-1] + shape[1:])
                    mps_conj[imps] = mps[imps].conj()
                    mps.qn[imps] = qnrset

                    rtensor = environ.GetLR(
                        "R", imps, mps, mps_conj, mpo, itensor=rtensor, method="System"
                    )
                    r_array = rtensor.array

                    # reverse update u site
                    shape_u = u.shape

                    def func_u(t, y):
                        return hop_svt(y.reshape(shape_u)).ravel() / 1.0j

                    sol_u = solve_ivp(func_u, (0, -evolve_dt / 2), u.ravel(), method="RK45")
                    rk_steps.append(len(sol_u.t))
                    mps[imps - 1] = tensordot(
                        mps[imps - 1].array, sol_u.y[:, -1].reshape(shape_u), axes=(-1, 0)
                    )
                    mps_conj[imps - 1] = mps[imps - 1].conj()

                elif mps.is_right_canon and imps != len(mps) - 1:
                    mps[imps] = u.reshape(shape[:-1] + [-1])
                    mps_conj[imps] = mps[imps].conj()
                    mps.qn[imps + 1] = qnlset

                    ltensor = environ.GetLR(
                        "L", imps, mps, mps_conj, mpo, itensor=ltensor, method="System"
                    )
                    l_array = ltensor.array

                    # reverse update svt site
                    shape_svt = vt.shape

                    def func_svt(t, y):
                        return hop_svt(y.reshape(shape_svt)).ravel() / 1.0j

                    sol_svt = solve_ivp(
                        func_svt, (0, -evolve_dt / 2), vt.ravel(), method="RK45"
                    )
                    rk_steps.append(len(sol_svt.t))
                    mps[imps + 1] = tensordot(
                        sol_svt.y[:, -1].reshape(shape_svt),
                        mps[imps + 1].array,
                        axes=(1, 0),
                    )
                    mps_conj[imps + 1] = mps[imps + 1].conj()

                else:
                    mps[imps] = mps_t
                    mps_conj[imps] = mps[imps].conj()
            mps._switch_domain()

        logger.debug(f"TDVP-PS RK steps: {stats.describe(rk_steps)}")

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
        mol_list = self.mol_list
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
            mpo_indep.scheme,
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

    def __str__(self):
        # too many digits in the default format
        e_occupations_str = ", ".join(
            ["%.2f" % number for number in self.e_occupations]
        )
        template_str = "current size: {}, peak size: {}, Matrix product bond order:{}, electron occupations: {}"
        return template_str.format(
            sizeof_fmt(self.total_bytes),
            sizeof_fmt(self.peak_bytes),
            self.bond_dims,
            e_occupations_str,
        )


def projector(ms: xp.ndarray) -> xp.ndarray:
    # projector
    proj = xp.tensordot(ms, ms.conj(), axes=(-1, -1))
    sz = int(np.prod(ms.shape[:-1]))
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


def integrand_func_factory(shape, hop, islast, S_inv: xp.ndarray):
    def func(t, y):
        y0 = y.reshape(shape)
        HC = hop(y0)
        if not islast:
            proj = projector(y0)
            if y0.ndim == 3:
                HC = tensordot(proj, HC, axes=([2, 3], [0, 1]))
                HC = tensordot(proj, HC, axes=([2, 3], [0, 1]))
            elif y0.ndim == 4:
                HC = tensordot(proj, HC, axes=([3, 4, 5], [0, 1, 2]))
                HC = tensordot(proj, HC, axes=([3, 4, 5], [0, 1, 2]))
        return tensordot(HC, S_inv, axes=(-1, 0)).ravel() / 1.0j

    return func


def transferMat(mps, mpsconj, domain, siteidx):
    """
    calculate the transfer matrix from the left hand or the right hand
    """
    val = np.ones([1, 1])
    if domain == "R":
        for imps in range(len(mps) - 1, siteidx - 1, -1):
            val = np.tensordot(mpsconj[imps], val, axes=(2, 0))
            val = np.tensordot(val, mps[imps], axes=([1, 2], [1, 2]))
    elif domain == "L":
        for imps in range(0, siteidx + 1, 1):
            val = np.tensordot(mpsconj[imps], val, axes=(0, 0))
            val = np.tensordot(val, mps[imps], axes=([0, 2], [1, 0]))

    return val
