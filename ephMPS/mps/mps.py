from __future__ import absolute_import, print_function, unicode_literals

import itertools
import logging
from functools import reduce

import numpy as np
from cached_property import cached_property
import scipy
from ephMPS.mps.tdh import unitary_propagation

from ephMPS.mps import svd_qn, rk
from ephMPS.mps.lib import updatemps
from ephMPS.mps.matrix import MatrixState
from ephMPS.mps.mp import MatrixProduct
from ephMPS.mps.mpo import Mpo
from ephMPS.mps.tdh import mflib
from ephMPS.utils import Quantity, OptimizeConfig, sizeof_fmt

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
        mps.qn = [[0], ]
        dim_list = [1, ]
        nmps = len(mpo)

        for imps in range(nmps - 1):

            # quantum number
            if mps.ephtable.is_electron(imps):
                # e site
                qnbig = list(itertools.chain.from_iterable([x, x + 1] for x in mps.qn[imps]))
            else:
                # ph site
                qnbig = list(itertools.chain.from_iterable([x] * mps.pbond_list[imps] for x in mps.qn[imps]))

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
            mt, mpsdim, mpsqn, nouse = updatemps(u_set, s_set, qnset, u_set, nexciton, m_max, percent=percent)
            # add the next mpsdim
            dim_list.append(mpsdim)
            mps.append(mt.reshape(dim_list[imps], mps.pbond_list[imps], dim_list[imps + 1]))
            mps.qn.append(mpsqn)

        # the last site
        mps.qn.append([0])
        dim_list.append(1)
        mps.append(np.random.random([dim_list[-2], mps.pbond_list[-1], dim_list[-1]]) - 0.5)

        mps.qnidx = len(mps) - 1
        mps.qntot = nexciton

        # print("self.dim", self.dim)
        mps._left_canon = True

        mps.wfn = []
        for mol in mps.mol_list:
            for ph in mol.hartree_phs:
                mps.wfn.append(np.random.random(ph.n_phys_dim))
        mps.wfn.append(1.0)

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

        mps.wfn = []

        for mol in mol_list:
            for ph in mol.hartree_phs:
                if max_entangled:
                    diag_elems = [1.0] * ph.n_phys_dim
                    mps.wfn.append(np.diag(diag_elems))
                else:
                    diag_elems = [1.0] + [0.0] * (ph.n_phys_dim - 1)
                    mps.wfn.append(np.array(diag_elems))
        # the coefficent a
        mps.wfn.append(1.0)

        mflib.normalize(mps.wfn)

        return mps

    def __init__(self):
        super(Mps, self).__init__()
        self.mtype = MatrixState
        self.wfn = [1]

        self.optimize_config = OptimizeConfig()

        self._prop_method = 'C_RK4'

        self.compress_add = False

    def to_complex(self, inplace=False):
        new_mp = super(Mps, self).to_complex(inplace=inplace)
        new_mp.wfn = [wfn.astype(np.complex128) for wfn in new_mp.wfn[:-1]] + [new_mp.wfn[-1]]
        return new_mp

    @property
    def nexciton(self):
        return self.qntot

    @property
    def prop_method(self):
        return self._prop_method

    @prop_method.setter
    def prop_method(self, value):
        assert value in rk.method_list
        self._prop_method = value

    @_cached_property
    def norm(self):
        # todo: get the fast version in the comment working
        ''' Fast version yet not safe. Needs further testing
        if self.is_left_canon:
            assert self.check_left_canonical()
            return np.linalg.norm(np.ravel(self[-1]))
        else:
            assert self.check_right_canonical()
            return np.linalg.norm(np.ravel(self[0]))
        '''
        return np.sqrt(self.conj().dot(self).real)

    def calc_e_occupation(self, idx):
        return self.expectation(Mpo.onsite(self.mol_list, 'a^\dagger a', mol_idx_set={idx}))

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
        for property in cached_property_set:
            if property in self.__dict__:
                del self.__dict__[property]

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
                new_mps[i] = np.zeros([mta.shape[0] + mtb.shape[0], pdim,
                                       mta.shape[2] + mtb.shape[2]], dtype=np.complex128)
                new_mps[i][:mta.shape[0], :, :mta.shape[2]] = mta[:, :, :]
                new_mps[i][mta.shape[0]:, :, mta.shape[2]:] = mtb[:, :, :]

            new_mps[-1] = np.vstack([self[-1], other[-1]])
        elif self.is_mpo:  # MPO
            new_mps[0] = np.concatenate((self[0], other[0]), axis=3)
            for i in range(1, self.site_num - 1):
                mta = self[i]
                mtb = other[i]
                pdimu = mta.shape[1]
                pdimd = mta.shape[2]
                assert pdimu == mtb.shape[1]
                assert pdimd == mtb.shape[2]

                new_mps[i] = np.zeros([mta.shape[0] + mtb.shape[0], pdimu, pdimd,
                                       mta.shape[3] + mtb.shape[3]], dtype=np.complex128)
                new_mps[i][:mta.shape[0], :, :, :mta.shape[3]] = mta[:, :, :, :]
                new_mps[i][mta.shape[0]:, :, :, mta.shape[3]:] = mtb[:, :, :, :]

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
    def normalize(self, norm=1.0):
        factor = norm / self.norm
        self.scale(factor, inplace=True)
        self.wfn[-1] *= norm
        return self

    def evolve(self, mpo=None, evolve_dt=None, norm=None, approx_eiht=None):
        if self.mol_list.pure_dmrg:
            return self.evolve_dmrg(mpo, evolve_dt, norm, approx_eiht)
        else:
            hybrid_mpo, HAM, Etot = self.construct_hybrid_Ham()
            mps = self.evolve_dmrg(hybrid_mpo, evolve_dt, norm, approx_eiht)
            mps.wfn = unitary_propagation(HAM, mps.wfn, Etot, evolve_dt)
            return mps

    # todo: separate the 2 methods (approx_eiht), because their parameters are orthogonal
    def evolve_dmrg(self, mpo=None, evolve_dt=None, norm=None, approx_eiht=None):
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
        if norm is not None:
            new_mps.normalize(norm)
        return new_mps


    def expectation(self, mpo):
        # todo: might cause performance problem when calculating a lot of expectations
        #       use dynamic programing to improve the performance.
        # todo: different bra and ket
        return self.conj().dot(mpo.apply(self)).real / self.norm ** 2

    @property
    def digest(self):
        if 10 < self.site_num:
            return None
        prod = np.eye(1).reshape(1, 1, 1)
        for ms in self:
            prod = np.tensordot(prod, ms, axes=1)
            prod = prod.reshape((prod.shape[0], -1, prod.shape[-1]))
        return {'var': prod.var(), 'mean': prod.mean(), 'ptp': prod.ptp()}

    def construct_hybrid_Ham(self, debug=False):
        '''
        construct hybrid DMRG and Hartree(-Fock) Hamiltonian
        '''
        mol_list = self.mol_list
        WFN = self.wfn
        nmols = len(mol_list)

        # many-body electronic part
        A_el = np.zeros((nmols))
        for imol in range(nmols):
            MPO = Mpo.onsite(mol_list, "a^\dagger a", dipole=False, mol_idx_set={imol})
            # A_el[imol] = mpslib.dot(mpslib.conj(MPS),mpslib.mapply(MPO,MPS)).real
            A_el[imol] = self.expectation(MPO)

        logger.info("dmrg_occ: %s" % A_el)

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
        MPO_indep = Mpo(mol_list)
        # e_mean = mpslib.dot(mpslib.conj(MPS),mpslib.mapply(MPO_indep,MPS))
        e_mean = self.expectation(MPO_indep)
        elocal_offset = np.array([mol_list[imol].hartree_e0 + B_vib_mol[imol] for imol in range(nmols)]).real
        e_mean += A_el.dot(elocal_offset)

        MPO = Mpo(mol_list, elocal_offset=elocal_offset, offset=Quantity(e_mean.real))

        Etot += e_mean

        iwfn = 0
        HAM = []
        for imol, mol in enumerate(mol_list):
            for iph, ph in enumerate(mol.hartree_phs):
                e_mean = mflib.exp_value(WFN[iwfn], ph.h_indep, WFN[iwfn])
                Etot += e_mean
                e_mean += A_el[imol] * B_vib[imol][iph]
                HAM.append(ph.h_indep + ph.h_dep * A_el[imol] - np.diag([e_mean] * WFN[iwfn].shape[0]))
                iwfn += 1
        logger.info("Etot= %g" % Etot)
        if debug:
            return MPO, HAM, Etot, A_el
        else:
            return MPO, HAM, Etot

    def hartree_wfn_diff(self, other):
        assert len(self.wfn) == len(other.wfn)
        res = []
        for wfn1, wfn2 in zip(self.wfn, other.wfn):
            res.append(scipy.linalg.norm(np.tensordot(wfn1, wfn1, axes=0) - np.tensordot(wfn2, wfn2, axes=0)))
        return np.array(res)

    def __str__(self):
        # too many digits in the default format
        e_occupations_str = ', '.join(['%.2f' % number for number in self.e_occupations])
        template_str = 'threshold: {:g}, current size: {}, peak size: {}, Matrix product bond order:{}, electron occupations: {}'
        return template_str.format(self.threshold, sizeof_fmt(self.total_bytes), sizeof_fmt(self.peak_bytes), self.bond_dims, e_occupations_str)
