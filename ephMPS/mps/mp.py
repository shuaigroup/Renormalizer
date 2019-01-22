# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
from __future__ import absolute_import, division

import copy
import inspect
import traceback
from functools import reduce
import logging

import numpy as np
import scipy

from ephMPS.mps import rk, svd_qn
from ephMPS.mps.matrix import Matrix
from ephMPS.utils import sizeof_fmt

logger = logging.getLogger(__name__)


class MatrixProduct(list):
    @classmethod
    def from_raw_list(cls, raw_list):
        new_mp = cls()
        if not raw_list:
            return new_mp
        new_mp.dtype = raw_list[0].dtype.type
        for mt in raw_list:
            new_mp.append(mt)
        return new_mp

    def __init__(self):
        super(MatrixProduct, self).__init__()
        self.mtype = Matrix
        self.dtype = np.float64

        self.mol_list = None
        self._ephtable = None

        # mpo also need to be compressed sometimes
        self._compress_method = 'svd'
        self.threshold = 1e-3

        self.peak_bytes = 0

        self.qn = None
        self.qnidx = None
        self.qntot = None

    @property
    def site_num(self):
        return len(self)

    @property
    def mol_num(self):
        return self.mol_list.mol_num

    @property
    def compress_method(self):
        return self._compress_method

    @compress_method.setter
    def compress_method(self, value):
        assert value in ['svd', 'variational']
        self._compress_method = value

    @property
    def is_mps(self):
        return self.mtype.is_ms

    @property
    def is_mpo(self):
        return self.mtype.is_mo

    @property
    def is_complex(self):
        return self.dtype == np.complex128

    @property
    def bond_dims(self):
        bond_dims = [mt.bond_dim[0] for mt in self] + [self[-1].bond_dim[-1]] if self.site_num else []
        return bond_dims

    @property
    def ephtable(self):
        if self._ephtable is not None:
            return self._ephtable
        else:
            return self.mol_list.ephtable

    @ephtable.setter
    def ephtable(self, ephtable):
        self._ephtable = ephtable

    @property
    def pbond_list(self):
        return self.mol_list.pbond_list

    def build_empty_qn(self):
        self.qntot = 0
        self.qnidx = 0
        self.qn = [[0] * dim for dim in self.bond_dims]

    def build_none_qn(self):
        self.qntot = None
        self.qnidx = None
        self.qn = None


    def move_qnidx(self, dstidx):
        """
        Quantum number has a boundary side, left hand of the side is L system qn,
        right hand of the side is R system qn, the sum of quantum number of L system
        and R system is tot.
        """
        # construct the L system qn
        for idx in range(self.qnidx + 1, len(self.qn) - 1):
            self.qn[idx] = [self.qntot - i for i in self.qn[idx]]

        # set boundary to fsite:
        for idx in range(len(self.qn) - 2, dstidx, -1):
            self.qn[idx] = [self.qntot - i for i in self.qn[idx]]
        self.qnidx = dstidx

    def check_left_canonical(self):
        """
        check L-canonical
        """
        for mt in self[:-1]:
            if not mt.check_lortho():
                return False
        return True

    def check_right_canonical(self):
        """
        check R-canonical
        """
        for mt in self[1:]:
            if not mt.check_rortho():
                return False
        return True

    @property
    def is_left_canon(self):
        return self.qnidx == self.site_num - 1

    @property
    def is_right_canon(self):
        return self.qnidx == 0

    @property
    def iter_idx_list(self):
        if self.is_left_canon:
            return range(self.site_num - 1, 0, -1)
        else:
            return range(0, self.site_num - 1)

    def _update_ms(self, idx, u, vt, sigma=None, qnlset=None, qnrset=None, m_trunc=None):
        if m_trunc is None:
            m_trunc = u.shape[1]
        u = u[:, :m_trunc]
        vt = vt[:m_trunc, :]
        if sigma is not None:
            sigma = sigma[:m_trunc]
            if self.is_left_canon:
                u = np.einsum('ji, i -> ji', u, sigma)
            else:
                vt = np.einsum('i, ij -> ij', sigma, vt)
        if self.is_left_canon:
            self[idx - 1] = np.tensordot(self[idx - 1], u, axes=1)
            ret_mpsi = np.reshape(vt, [m_trunc] + list(self[idx].pdim) + [vt.shape[1] // self[idx].pdim_prod])
            if qnrset is not None:
                self.qn[idx] = qnrset[:m_trunc]
        else:
            self[idx + 1] = np.tensordot(vt, self[idx + 1], axes=1)
            ret_mpsi = np.reshape(u, [u.shape[0] // self[idx].pdim_prod] + list(self[idx].pdim) + [m_trunc])
            if qnlset is not None:
                self.qn[idx + 1] = qnlset[:m_trunc]
        assert ret_mpsi.any()
        self[idx] = ret_mpsi

    def _switch_domain(self):
        if self.is_left_canon:
            self.qnidx = 0
        else:
            self.qnidx = self.site_num - 1

    def _get_big_qn(self, idx):
        mt = self[idx]
        if self.ephtable.is_electron(idx):
            # e site
            sigmaqn = mt.elec_sigmaqn
        else:
            # ph site
            sigmaqn = np.array([0] * mt.pdim_prod)
        qnl = np.array(self.qn[idx])
        qnr = np.array(self.qn[idx + 1])
        if self.is_left_canon:
            qnbigl = qnl
            qnbigr = np.add.outer(sigmaqn, qnr)
        else:
            qnbigl = np.add.outer(qnl, sigmaqn)
            qnbigr = qnr
        return qnbigl, qnbigr

    def compress(self, check_canonical=True):
        """
        inp: canonicalise MPS (or MPO)

        trunc=0: just canonicalise
        0<trunc<1: sigma threshold
        trunc>1: number of renormalised vectors to keep

        side='l': compress LEFT-canonicalised MPS
                  by sweeping from RIGHT to LEFT
                  output MPS is right canonicalised i.e. CRRR

        side='r': reverse of 'l'

        returns:
             truncated or canonicalised MPS
        """

        # ensure mps is canonicalised
        if check_canonical:
            if self.is_left_canon:
                assert self.check_left_canonical()
            else:
                assert self.check_right_canonical()
        system = 'R' if self.is_left_canon else 'L'

        for idx in self.iter_idx_list:
            mt = self[idx]
            assert mt.any()
            if self.is_left_canon:
                mt = mt.r_combine()
            else:
                mt = mt.l_combine()
            qnbigl, qnbigr = self._get_big_qn(idx)
            u, sigma, qnlset, v, sigma, qnrset = svd_qn.Csvd(mt, qnbigl, qnbigr, self.qntot,
                                                             system=system, full_matrices=False)
            vt = v.T

            if self.threshold < 1.:
                # count how many sing vals < trunc
                normed_sigma = sigma / scipy.linalg.norm(sigma)
                # m_trunc=len([s for s in normed_sigma if s >trunc])
                m_trunc = np.count_nonzero(normed_sigma > self.threshold)
            else:
                assert False # in some cases buggy, such as dynamic threshold
                # m_trunc = int(self.threshold)
                # m_trunc = min(m_trunc, len(sigma))
            assert m_trunc != 0
            self._update_ms(idx, u, vt, sigma, qnlset, qnrset, m_trunc)

        self._switch_domain()

    def canonicalise(self):
        for idx in self.iter_idx_list:
            mt = self[idx]
            assert mt.any()
            if self.is_left_canon:
                mt = mt.r_combine()
            else:
                mt = mt.l_combine()
            qnbigl, qnbigr = self._get_big_qn(idx)
            system = 'R' if self.is_left_canon else 'L'
            u, qnlset, v, qnrset = svd_qn.Csvd(mt, qnbigl, qnbigr, self.qntot,
                                               QR=True, system=system, full_matrices=False)
            vt = v.T
            self._update_ms(idx, u, vt, sigma=None, qnlset=qnlset, qnrset=qnrset)
        self._switch_domain()

    def conj(self):
        """
        complex conjugate
        """
        new_mp = self.copy()
        for idx, mt in enumerate(new_mp):
            new_mp[idx] = mt.conj()
        return new_mp

    def dot(self, other):
        """
        dot product of two mps / mpo 
        """

        assert len(self) == len(other)
        e0 = np.eye(1, 1)
        for mt1, mt2 in zip(self, other):
            # sum_x e0[:,x].m[x,:,:]
            e0 = np.tensordot(e0, mt2, 1)
            # sum_ij e0[i,p,:] self[i,p,:]
            # note, need to flip a (:) index onto top,
            # therefore take transpose
            if mt1.ndim == 3:
                e0 = np.tensordot(e0, mt1, ([0, 1], [0, 1])).T
            if mt2.ndim == 4:
                e0 = np.tensordot(e0, mt1, ([0, 1, 2], [0, 1, 2])).T

        return e0[0, 0]

    def angle(self, other):
        return np.abs(self.conj().dot(other))

    def scale(self, val, inplace=False):
        new_mp = self if inplace else self.copy()
        if np.iscomplexobj(val):
            new_mp.to_complex(inplace=True)
        new_mp[self.qnidx] *= val
        return new_mp

    def to_complex(self, inplace=False):
        if inplace:
            new_mp = self
        else:
            new_mp = self.copy()
        new_mp.dtype = np.complex128
        for i in range(new_mp.site_num):
            new_mp[i] = new_mp.dtype(new_mp[i])
        return new_mp

    def to_raw_list(self):
        return [np.array(mt) for mt in self]

    def distance(self, other):
        if not hasattr(other, 'conj'):
            other = self.__class__.from_raw_list(other)
        return self.conj().dot(self) - np.abs(self.conj().dot(other)) - np.abs(
            other.conj().dot(self)) + other.conj().dot(other)

    def copy(self):
        return copy.deepcopy(self)

    def array2mt(self, array):
        return self.mtype(self.dtype(array))

    @property
    def total_bytes(self):
        return sum([array.nbytes for array in self])

    def set_peak_bytes(self, new_bytes=None):
        if new_bytes is None:
            new_bytes = self.total_bytes
        if new_bytes < self.peak_bytes:
            return
        self.peak_bytes = new_bytes
        stack = ''.join(traceback.format_stack(inspect.stack()[2].frame, 1)).replace('\n', ' ')
        logger.debug('Set peak bytes to {}. Called from: {}'.format(sizeof_fmt(new_bytes), stack))

    def __eq__(self, other):
        for m1, m2 in zip(self, other):
            if not np.allclose(m1, m2):
                return False
        return True

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return '%s with %d sites' % (self.__class__, len(self))

    def __setitem__(self, key, array):
        super(MatrixProduct, self).__setitem__(key, self.array2mt(array))

    def append(self, array):
        super(MatrixProduct, self).append(self.array2mt(array))
