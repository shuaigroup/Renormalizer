from __future__ import absolute_import, print_function, unicode_literals

import logging
from functools import reduce

import numpy as np
import scipy
from cached_property import cached_property


from ephMPS.mps.tdh import unitary_propagation
from ephMPS.mps import svd_qn, rk
from ephMPS.mps.lib import construct_enviro, GetLR, updatemps, transferMat
from ephMPS.mps.mp import MatrixProduct
from ephMPS.mps.mpo import Mpo
from ephMPS.mps.tdh import mflib
from ephMPS.lib import tensor as tensorlib
from ephMPS.lib import solve_ivp
from ephMPS.utils import (
    Quantity,
    OptimizeConfig,
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
    def wrapper(self, *args, **kwargs):
        ret = f(self, *args, **kwargs)
        assert isinstance(ret, self.__class__)
        ret.invalidate_cache()
        return ret

    return wrapper


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
                mt.reshape(dim_list[imps], mps.pbond_list[imps], dim_list[imps + 1])
            )
            mps.qn.append(mpsqn)

        # the last site
        mps.qn.append([0])
        dim_list.append(1)
        mps.append(
            np.random.random([dim_list[-2], mps.pbond_list[-1], dim_list[-1]]) - 0.5
        )

        mps.qnidx = len(mps) - 1
        mps.qntot = nexciton

        # print("self.dim", self.dim)
        mps._left_canon = True

        mps.wfns = []
        for mol in mps.mol_list:
            for ph in mol.hartree_phs:
                mps.wfns.append(np.random.random(ph.n_phys_dim))
        mps.wfns.append(1.0)

        return mps

    @classmethod
    def gs(cls, mol_list, max_entangled):
        """
        T = \infty maximum entangled GS state
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

        self.optimize_config = OptimizeConfig()
        self.evolve_config = EvolveConfig()

        self.compress_add = False

    def conj(self):
        new_mps = super(Mps, self).conj()
        for idx, wfn in enumerate(new_mps.wfns):
            new_mps.wfns[idx] = np.conj(wfn)
        return new_mps

    def dot(self, other, with_hartree=True):
        e = super(Mps, self).dot(other)
        if with_hartree:
            assert len(self.wfns) == len(other.wfns)
            for wfn1, wfn2 in zip(self.wfns[:-1], other.wfns[:-1]):
                # use vdot is buggy here, because vdot will take conjugation automatically
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
    def nexciton(self):
        return self.qntot

    @property
    def prop_method(self):
        return self.evolve_config.prop_method

    @prop_method.setter
    def prop_method(self, value):
        assert value in rk.method_list
        self.evolve_config.prop_method = value

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

    def calc_e_occupation(self, idx):
        return self.expectation(
            Mpo.onsite(self.mol_list, "a^\dagger a", mol_idx_set={idx})
        )

    def calc_ph_occupation(self, mol_idx, ph_idx):
        return self.expectation(Mpo.ph_occupation_mpo(self.mol_list, mol_idx, ph_idx))

    @_cached_property
    def ph_occupations(self):
        ph_occupations = []
        for imol, mol in enumerate(self.mol_list):
            for iph in range(len(mol.dmrg_phs)):
                ph_occupations.append(self.calc_ph_occupation(imol, iph))
        return np.array(ph_occupations)

    @_cached_property
    def e_occupations(self):
        return np.array([self.calc_e_occupation(i) for i in range(self.mol_num)])

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

    @invalidate_cache_decorator
    def copy(self):
        return super(Mps, self).copy()

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

        new_mps = other.copy()
        new_mps.threshold = min(self.threshold, other.threshold)

        if self.is_mps:  # MPS
            new_mps[0] = np.dstack([self[0], other[0]])
            for i in range(1, self.site_num - 1):
                mta = self[i]
                mtb = other[i]
                pdim = mta.shape[1]
                assert pdim == mtb.shape[1]
                new_mps[i] = np.zeros(
                    [mta.shape[0] + mtb.shape[0], pdim, mta.shape[2] + mtb.shape[2]],
                    dtype=np.complex128,
                )
                new_mps[i][: mta.shape[0], :, : mta.shape[2]] = mta[:, :, :]
                new_mps[i][mta.shape[0] :, :, mta.shape[2] :] = mtb[:, :, :]

            new_mps[-1] = np.vstack([self[-1], other[-1]])
        elif self.is_mpo or self.is_mpdm:  # MPO
            new_mps[0] = np.concatenate((self[0], other[0]), axis=3)
            for i in range(1, self.site_num - 1):
                mta = self[i]
                mtb = other[i]
                pdimu = mta.shape[1]
                pdimd = mta.shape[2]
                assert pdimu == mtb.shape[1]
                assert pdimd == mtb.shape[2]

                new_mps[i] = np.zeros(
                    [
                        mta.shape[0] + mtb.shape[0],
                        pdimu,
                        pdimd,
                        mta.shape[3] + mtb.shape[3],
                    ],
                    dtype=np.complex128,
                )
                new_mps[i][: mta.shape[0], :, :, : mta.shape[3]] = mta[:, :, :, :]
                new_mps[i][mta.shape[0] :, :, :, mta.shape[3] :] = mtb[:, :, :, :]

                new_mps[-1] = np.concatenate((self[-1], other[-1]), axis=0)
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

    def canonical_normalize(self):
        # applied by a operator then normalize: dmrg should be normalized,
        #   tdh should be normalized, coefficient is set to the length
        # suppose length is only determined by dmrg part
        self.normalize(self.dmrg_norm)

    def evolve(self, mpo=None, evolve_dt=None, approx_eiht=None):
        hybrid_mpo, HAM, Etot = self.construct_hybrid_Ham(mpo)
        mps = self.evolve_dmrg(hybrid_mpo, evolve_dt, approx_eiht)
        unitary_propagation(mps.wfns, HAM, Etot, evolve_dt)
        if np.iscomplex(evolve_dt):
            mps.normalize(1.0)
        else:
            mps.normalize(None)
        return mps

    # todo: separate the 2 methods (approx_eiht), because their parameters are orthogonal
    def evolve_dmrg(self, mpo=None, evolve_dt=None, approx_eiht=None):
        if self.evolve_config.scheme == EvolveMethod.prop_and_compress:
            return self.evolve_dmrg_prop_and_compress(mpo, evolve_dt, approx_eiht)
        if self.evolve_config.scheme == EvolveMethod.tdvp_mctdh:
            return self.evolve_dmrg_tdvp_mctdh(mpo, evolve_dt)
        elif self.evolve_config.scheme == EvolveMethod.tdvp_mctdh_new:
            return self.evolve_dmrg_tdvp_mctdhnew(mpo, evolve_dt)
        elif self.evolve_config.scheme == EvolveMethod.tdvp_ps:
            return self.evolve_dmrg_tdvp_ps(mpo, evolve_dt)
        else:
            assert False

    def evolve_dmrg_prop_and_compress(self, mpo=None, evolve_dt=None, approx_eiht=None):
        if approx_eiht is not None:
            return approx_eiht.contract(self)
        propagation_c = rk.coefficient_dict[self.prop_method]
        termlist = [self]
        while len(termlist) < len(propagation_c):
            termlist.append(mpo.contract(termlist[-1]))
            # control term sizes to be approximately constant
            termlist[-1].threshold *= 3
        for idx, term in enumerate(termlist):
            term.scale((-1.0j * evolve_dt) ** idx * propagation_c[idx], inplace=True)
        new_mps = reduce(lambda mps1, mps2: mps1.add(mps2), termlist)
        if not self.compress_add:
            new_mps.canonicalise()
            new_mps.compress()
        return new_mps

    def evolve_dmrg_tdvp_mctdh(self, mpo, evolve_dt):
        # TDVP for original MCTDH
        if self.is_left_canon:
            assert self.check_left_canonical()
            self.canonicalise()
        # qn for this method has not been implemented
        self.use_dummy_qn = True
        self.clear_qn()
        mps = self.to_complex(inplace=True)
        mps_conj = mps.conj()
        construct_enviro(mps, mps_conj, mpo, "R")

        # initial matrix
        ltensor = np.ones((1, 1, 1))
        rtensor = np.ones((1, 1, 1))

        new_mps = self.copy()

        for imps in range(len(mps)):
            ltensor = GetLR(
                "L", imps - 1, mps, mps_conj, mpo, itensor=ltensor, method="System"
            )
            rtensor = GetLR(
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
            S_inv = u.dot(np.diag(1.0 / w)).dot(np.conj(u.T))

            # pseudo inverse
            # S_inv = scipy.linalg.pinvh(S,rcond=1e-2)

            hop = hop_factory(ltensor, rtensor, mpo[imps])

            shape = mps[imps].shape

            func = integrand_func_factory(shape, hop, imps == len(mps) - 1, S_inv)

            sol = solve_ivp(func, (0, evolve_dt), mps[imps].ravel(), method="RK45")
            # print
            # "CMF steps:", len(sol.t)
            new_mps[imps] = sol.y[:, -1].reshape(shape)
            # print
            # "orthogonal1", np.allclose(np.tensordot(MPSnew[imps],
            #                                        np.conj(MPSnew[imps]), axes=([0, 1], [0, 1])),
            #                           np.diag(np.ones(MPSnew[imps].shape[2])))

        mps._switch_domain()
        new_mps._switch_domain()
        new_mps.canonicalise()
        return new_mps

    def evolve_dmrg_tdvp_mctdhnew(self, mpo, evolve_dt):
        # new regularization scheme
        # JCP 148, 124105 (2018)
        # JCP 149, 044119 (2018)

        if self.is_left_canon:
            assert self.check_left_canonical()
            self.canonicalise()
        # qn for this method has not been implemented
        self.use_dummy_qn = True
        self.clear_qn()
        # todo: use 4x memory here, could be reduced to 2x
        mps = self.to_complex(inplace=True)

        # construct the environment matrix
        construct_enviro(mps, mps.conj(), mpo, "R")

        # initial matrix
        ltensor = np.ones((1, 1, 1))
        rtensor = np.ones((1, 1, 1))

        new_mps = mps.copy()

        for imps in range(len(mps)):
            shape = list(mps[imps].shape)

            u, s, vt = scipy.linalg.svd(
                mps[imps].reshape(-1, shape[-1]), full_matrices=False
            )
            mps[imps] = u.reshape(shape[:-1] + [-1])

            ltensor = GetLR(
                "L", imps - 1, mps, mps.conj(), mpo, itensor=ltensor, method="System"
            )
            rtensor = GetLR(
                "R", imps + 1, mps, mps.conj(), mpo, itensor=rtensor, method="Enviro"
            )

            epsilon = 1e-10
            epsilon = np.sqrt(epsilon)
            s = s + epsilon * np.exp(-s / epsilon)

            svt = np.diag(s).dot(vt)

            rtensor = np.tensordot(rtensor, svt, axes=(2, 1))
            rtensor = np.tensordot(np.conj(vt), rtensor, axes=(1, 0))

            if imps != len(mps) - 1:
                mps[imps + 1] = np.tensordot(svt, mps[imps + 1], axes=(-1, 0))

            # density matrix
            S = s * s
            # print
            # "sum density matrix", np.sum(S)

            S_inv = np.diag(1.0 / s)

            hop = hop_factory(ltensor, rtensor, mpo[imps])

            func = integrand_func_factory(shape, hop, imps == len(mps) - 1, S_inv)

            sol = solve_ivp(func, (0, evolve_dt), mps[imps].ravel(), method="RK45")
            # print
            # "CMF steps:", len(sol.t)
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

    def evolve_dmrg_tdvp_ps(self, mpo, evolve_dt):
        # TDVP projector splitting
        if self.is_right_canon:
            assert self.check_right_canonical()
            self.canonicalise()
        # qn for this method has not been implemented
        self.use_dummy_qn = True
        self.clear_qn()
        mps = self.to_complex()

        # construct the environment matrix
        construct_enviro(mps, mps.conj(), mpo, "L")

        # initial matrix
        ltensor = np.ones((1, 1, 1))
        rtensor = np.ones((1, 1, 1))

        loop = [["R", i] for i in range(len(mps) - 1, -1, -1)] + [
            ["L", i] for i in range(0, len(mps))
        ]
        for system, imps in loop:
            if system == "R":
                lmethod, rmethod = "Enviro", "System"
                ltensor = GetLR(
                    "L", imps - 1, mps, mps.conj(), mpo, itensor=ltensor, method=lmethod
                )
            else:
                lmethod, rmethod = "System", "Enviro"
                rtensor = GetLR(
                    "R", imps + 1, mps, mps.conj(), mpo, itensor=rtensor, method=rmethod
                )

            hop = hop_factory(ltensor, rtensor, mpo[imps])

            def hop_svt(mps):
                # S-a   l-S
                #
                # O-b - b-O
                #
                # S-c   k-S

                path = [([0, 1], "abc, ck -> abk"), ([1, 0], "abk, lbk -> al")]
                HC = tensorlib.multi_tensor_contract(path, ltensor, mps, rtensor)
                return HC

            shape = list(mps[imps].shape)

            def func(t, y):
                return hop(y.reshape(shape)).ravel() / 1.0j

            sol = solve_ivp(
                func, (0, evolve_dt / 2.0), mps[imps].ravel(), method="RK45"
            )
            # print
            # "nsteps for MPS[imps]:", len(sol.t)
            mps_t = sol.y[:, -1].reshape(shape)

            if system == "L" and imps != len(mps) - 1:
                # updated imps site
                u, vt = scipy.linalg.qr(mps_t.reshape(-1, shape[-1]), mode="economic")
                mps[imps] = u.reshape(shape[:-1] + [-1])

                ltensor = GetLR(
                    "L", imps, mps, mps.conj(), mpo, itensor=ltensor, method="System"
                )

                # reverse update svt site
                shape_svt = vt.shape

                def func_svt(t, y):
                    return hop_svt(y.reshape(shape_svt)).ravel() / 1.0j

                sol_svt = solve_ivp(
                    func_svt, (0, -evolve_dt / 2), vt.ravel(), method="RK45"
                )
                # print
                # "nsteps for svt:", len(sol_svt.t)
                mps[imps + 1] = np.tensordot(
                    sol_svt.y[:, -1].reshape(shape_svt), mps[imps + 1], axes=(1, 0)
                )

            elif system == "R" and imps != 0:
                # updated imps site
                u, vt = scipy.linalg.rq(mps_t.reshape(shape[0], -1), mode="economic")
                mps[imps] = vt.reshape([-1] + shape[1:])

                rtensor = GetLR(
                    "R", imps, mps, mps.conj(), mpo, itensor=rtensor, method="System"
                )

                # reverse update u site
                shape_u = u.shape

                def func_u(t, y):
                    return hop_svt(y.reshape(shape_u)).ravel() / 1.0j

                sol_u = solve_ivp(func_u, (0, -evolve_dt / 2), u.ravel(), method="RK45")
                # print
                # "nsteps for u:", len(sol_u.t)
                mps[imps - 1] = np.tensordot(
                    mps[imps - 1], sol_u.y[:, -1].reshape(shape_u), axes=(-1, 0)
                )

            else:
                mps[imps] = mps_t

        return mps

        # print
        # "tMPS dim:", [mps.shape[0] for mps in MPSnew] + [1]

    def evolve_exact(self, h_mpo, evolve_dt, space):
        MPOprop, HAM, Etot = self.hybrid_exact_propagator(
            h_mpo, -1.0j * evolve_dt, space
        )
        new_mps = MPOprop.apply(self, canonicalise=True)
        unitary_propagation(new_mps.wfns, HAM, Etot, evolve_dt)
        return new_mps

    def expectation(self, mpo):
        # todo: might cause performance problem when calculating a lot of expectations
        #       use dynamic programing to improve the performance.
        # todo: different bra and ket
        return self.conj().dot(mpo.apply(self), with_hartree=False).real

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
        # e_mean = mpslib.dot(mpslib.conj(MPS),mpslib.mapply(MPO_indep,MPS))
        e_mean = self.expectation(mpo_indep)
        elocal_offset = np.array(
            [mol_list[imol].hartree_e0 + B_vib_mol[imol] for imol in range(nmols)]
        ).real
        e_mean += A_el.dot(elocal_offset)
        total_offset = mpo_indep.offset + Quantity(e_mean.real)
        MPO = Mpo(mol_list, elocal_offset=elocal_offset, offset=total_offset)

        Etot += e_mean

        iwfn = 0
        HAM = []
        for imol, mol in enumerate(mol_list):
            for iph, ph in enumerate(mol.hartree_phs):
                e_mean = mflib.exp_value(WFN[iwfn], ph.h_indep, WFN[iwfn])
                Etot += e_mean
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
        template_str = "threshold: {:g}, current size: {}, peak size: {}, Matrix product bond order:{}, electron occupations: {}"
        return template_str.format(
            self.threshold,
            sizeof_fmt(self.total_bytes),
            sizeof_fmt(self.peak_bytes),
            self.bond_dims,
            e_occupations_str,
        )


def projector(ms):
    # projector
    proj = np.tensordot(ms, np.conj(ms), axes=(-1, -1))
    Iden = np.diag(np.ones(np.prod(ms.shape[:-1]))).reshape(proj.shape)
    proj = Iden - proj
    return proj


def hop_factory(ltensor, rtensor, mo):
    def hop(ms):
        # S-a   l-S
        #     d
        # O-b-O-f-O
        #     e
        # S-c   k-S
        if ms.ndim == 3:
            path = [
                ([0, 1], "abc, cek -> abek"),
                ([2, 0], "abek, bdef -> akdf"),
                ([1, 0], "akdf, lfk -> adl"),
            ]
            HC = tensorlib.multi_tensor_contract(path, ltensor, ms, mo, rtensor)

        # S-a   l-S
        #     d
        # O-b-O-f-O
        #     e
        # S-c   k-S
        #     g
        elif ms.ndim == 4:
            path = [
                ([0, 1], "abc, bdef -> acdef"),
                ([2, 0], "acdef, cegk -> adfgk"),
                ([1, 0], "adfgk, lfk -> adgl"),
            ]
            HC = tensorlib.multi_tensor_contract(path, ltensor, mo, ms, rtensor)
        else:
            assert False
        return HC

    return hop


def integrand_func_factory(shape, hop, islast, S_inv):
    def func(t, y):
        y0 = y.reshape(shape)
        HC = hop(y0)
        if not islast:
            proj = projector(y0)
            if y0.ndim == 3:
                HC = np.tensordot(proj, HC, axes=([2, 3], [0, 1]))
                HC = np.tensordot(proj, HC, axes=([2, 3], [0, 1]))
            elif y0.ndim == 4:
                HC = np.tensordot(proj, HC, axes=([3, 4, 5], [0, 1, 2]))
                HC = np.tensordot(proj, HC, axes=([3, 4, 5], [0, 1, 2]))
        return np.tensordot(HC, S_inv, axes=(-1, 0)).ravel() / 1.0j

    return func
