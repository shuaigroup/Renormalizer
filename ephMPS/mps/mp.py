# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
from __future__ import absolute_import, division

import inspect
import traceback
import logging
from typing import List, Union

import numpy as np
import scipy

from ephMPS.mps.matrix import Matrix, backend, eye, einsum, tensordot, allclose
from ephMPS.mps import svd_qn
from ephMPS.model import MolList, EphTable
from ephMPS.utils import sizeof_fmt, CompressConfig

logger = logging.getLogger(__name__)


class MatrixProduct:
    @classmethod
    def from_raw_list(cls, raw_list, mol_list):
        new_mp = cls()
        new_mp.mol_list = mol_list
        if not raw_list:
            return new_mp
        new_mp.dtype = raw_list[0].dtype.type
        for mt in raw_list:
            new_mp.append(mt)
        return new_mp

    def __init__(self):
        # when modify theses codes, keep in mind to update `metacopy` method
        # set to a list of None upon metacopy
        self._mp: List[Union[Matrix, None]] = []
        self.dtype = backend.real_dtype

        # in mpo.quasi_boson, mol_list is not set, then _ephtable and _pbond_list should be used
        self.mol_list: MolList = None
        self._ephtable: EphTable = None
        self._pbond_list: List[int] = None

        # mpo also need to be compressed sometimes
        self.compress_config: CompressConfig = CompressConfig()

        # maximum size during the whole life time of the mp
        self.peak_bytes: int = 0

        # QN related
        self.use_dummy_qn: bool = False
        # self.use_dummy_qn = True
        self.qn: List[List[int]] = []
        self.qnidx: int = None
        self._qntot: int = None

    @property
    def site_num(self):
        return len(self._mp)

    @property
    def mol_num(self):
        return self.mol_list.mol_num

    @property
    def threshold(self):
        return self.compress_config.threshold

    @threshold.setter
    def threshold(self, v):
        self.compress_config.threshold = v

    @property
    def is_mps(self):
        raise NotImplementedError

    @property
    def is_mpo(self):
        raise NotImplementedError

    @property
    def is_mpdm(self):
        raise NotImplementedError

    @property
    def is_complex(self):
        return self.dtype == backend.complex_dtype

    @property
    def bond_dims(self) -> np.ndarray:
        bond_dims = (
            [mt.bond_dim[0] for mt in self] + [self[-1].bond_dim[-1]]
            if self.site_num
            else []
        )
        return np.array(bond_dims)

    @property
    def ephtable(self):
        if self._ephtable is not None:
            return self._ephtable
        else:
            return self.mol_list.ephtable

    @ephtable.setter
    def ephtable(self, ephtable: EphTable):
        self._ephtable = ephtable

    @property
    def pbond_list(self):
        if self._pbond_list is not None:
            return self._pbond_list
        else:
            return self.mol_list.pbond_list

    @pbond_list.setter
    def pbond_list(self, pbond_list):
        self._pbond_list = pbond_list

    @property
    def qntot(self):
        if self.use_dummy_qn:
            return 0
        else:
            return self._qntot

    @qntot.setter
    def qntot(self, qntot: int):
        if not self.use_dummy_qn:
            self._qntot = qntot

    def build_empty_qn(self):
        self.qntot = 0
        self.qnidx = 0
        self.qn = [[0] * dim for dim in self.bond_dims]

    def build_none_qn(self):
        self.qntot = None
        self.qnidx = None
        self.qn = None

    def clear_qn(self):
        self.qntot = 0
        self.qn = [[0] * dim for dim in self.bond_dims]

    def move_qnidx(self, dstidx: int):
        """
        Quantum number has a boundary site, left hand of the site is L system qn,
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
        # Note that this function doesn't mean all sites. The last is omitted.
        if self.is_left_canon:
            return range(self.site_num - 1, 0, -1)
        else:
            return range(0, self.site_num - 1)

    def _update_ms(
        self, idx, u, vt, sigma=None, qnlset=None, qnrset=None, m_trunc=None
    ):
        if m_trunc is None:
            m_trunc = u.shape[1]
        u = u[:, :m_trunc]
        vt = vt[:m_trunc, :]
        if sigma is not None:
            sigma = sigma[:m_trunc]
            if self.is_left_canon:
                u = einsum("ji, i -> ji", u, sigma)
            else:
                vt = einsum("i, ij -> ij", sigma, vt)
        if self.is_left_canon:
            self[idx - 1] = tensordot(self[idx - 1], u, axes=1)
            ret_mpsi = vt.reshape(
                [m_trunc] + list(self[idx].pdim) + [vt.shape[1] // self[idx].pdim_prod]
            )
            if qnrset is not None:
                self.qn[idx] = qnrset[:m_trunc]
        else:
            self[idx + 1] = tensordot(vt, self[idx + 1], axes=1)
            ret_mpsi = u.reshape(
                [u.shape[0] // self[idx].pdim_prod] + list(self[idx].pdim) + [m_trunc]
            )
            if qnlset is not None:
                self.qn[idx + 1] = qnlset[:m_trunc]
        if ret_mpsi.nbytes < ret_mpsi.base.nbytes * 0.8:
            # do copy here to discard unnecessary data. Note that in NumPy common slicing returns
            # a `view` containing the original data. If `ret_mpsi` is used directly the original
            # `u` or `vt` is not garbage collected.
            ret_mpsi = ret_mpsi.copy()
        assert ret_mpsi.any()
        self[idx] = ret_mpsi

    def _switch_domain(self):
        if self.is_left_canon:
            self.qnidx = 0
            # assert self.check_right_canonical()
        else:
            self.qnidx = self.site_num - 1
            # assert self.check_left_canonical()

    def _get_big_qn(self, idx):
        mt: Matrix = self[idx]
        sigmaqn = mt.sigmaqn
        qnl = np.array(self.qn[idx])
        qnr = np.array(self.qn[idx + 1])
        assert len(qnl) == mt.shape[0]
        assert len(qnr) == mt.shape[-1]
        assert len(sigmaqn) == mt.pdim_prod
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
        system = "R" if self.is_left_canon else "L"

        for idx in self.iter_idx_list:
            mt: Matrix = self[idx]
            assert mt.any()
            if self.is_left_canon:
                mt = mt.r_combine()
            else:
                mt = mt.l_combine()
            qnbigl, qnbigr = self._get_big_qn(idx)
            u, sigma, qnlset, v, sigma, qnrset = svd_qn.Csvd(
                mt.asnumpy(),
                qnbigl,
                qnbigr,
                self.qntot,
                system=system,
                full_matrices=False,
            )
            vt = v.T
            m_trunc = self.compress_config.compute_m_trunc(
                sigma, idx, self.is_left_canon
            )
            self._update_ms(
                idx, Matrix(u), Matrix(vt), Matrix(sigma), qnlset, qnrset, m_trunc
            )

        self._switch_domain()

    def canonicalise(self):
        for idx in self.iter_idx_list:
            mt: Matrix = self[idx]
            assert mt.any()
            if self.is_left_canon:
                mt = mt.r_combine()
            else:
                mt = mt.l_combine()
            qnbigl, qnbigr = self._get_big_qn(idx)
            system = "R" if self.is_left_canon else "L"
            u, qnlset, v, qnrset = svd_qn.Csvd(
                mt.asnumpy(),
                qnbigl,
                qnbigr,
                self.qntot,
                QR=True,
                system=system,
                full_matrices=False,
            )
            self._update_ms(
                idx, Matrix(u), Matrix(v.T), sigma=None, qnlset=qnlset, qnrset=qnrset
            )
        self._switch_domain()

    def conj(self):
        """
        complex conjugate
        """
        new_mp = self.metacopy()
        for idx, mt in enumerate(self):
            new_mp[idx] = mt.conj()
        return new_mp

    def dot(self, other):
        """
        dot product of two mps / mpo 
        """

        assert len(self) == len(other)
        e0 = eye(1, 1)
        for mt1, mt2 in zip(self, other):
            # sum_x e0[:,x].m[x,:,:]
            e0 = tensordot(e0, mt2, 1)
            # sum_ij e0[i,p,:] self[i,p,:]
            # note, need to flip a (:) index onto top,
            # therefore take transpose
            if mt1.ndim == 3:
                e0 = tensordot(e0, mt1, ([0, 1], [0, 1])).T
            elif mt1.ndim == 4:
                e0 = tensordot(e0, mt1, ([0, 1, 2], [0, 1, 2])).T
            else:
                assert False

        return e0[0, 0]

    def angle(self, other):
        return abs(self.conj().dot(other))

    def scale(self, val, inplace=False):
        new_mp = self if inplace else self.copy()
        if np.iscomplexobj(val):
            new_mp.to_complex(inplace=True)
        # Note matrices are read-only
        # there are two ways to do the scaling
        if np.abs(np.log(np.abs(val))) < 0.01:
            # Thr first way. The operation performs very quickly,
            # but leads to high float point error when val is very large or small
            new_mp[self.qnidx] = new_mp[self.qnidx] * val
        else:
            # The second way. High time complexity but numerically more feasible.
            root_val = val ** (1 / len(self))
            for idx, mt in enumerate(self):
                new_mp[idx] = mt * root_val
        # the two ways could be united. I'm currently not confident enough that
        # the modification will work. So explicitly use two ways for now
        return new_mp

    def to_complex(self, inplace=False):
        if inplace:
            new_mp = self
        else:
            new_mp = self.metacopy()
        new_mp.dtype = backend.complex_dtype
        for i, mt in enumerate(self):
            new_mp[i] = mt.to_complex(inplace)
        return new_mp

    def distance(self, other):
        if not hasattr(other, "conj"):
            other = self.__class__.from_raw_list(other, self.mol_list)
        return (
            self.conj().dot(self)
            - abs(self.conj().dot(other))
            - abs(other.conj().dot(self))
            + other.conj().dot(other)
        )

    def copy(self):
        new = self.metacopy()
        new._mp = [m.copy() for m in self._mp]
        return new

    # only (shalow) copy metadata because usually after been copied the real data is overwritten
    def metacopy(self) -> "MatrixProduct":
        new = self.__class__.__new__(self.__class__)
        new._mp = [None] * len(self)
        new.dtype = self.dtype
        new.mol_list = self.mol_list
        new._ephtable = self._ephtable
        new._pbond_list = self._pbond_list
        # need to deep copy compress_config because threshold might change dynamically
        new.compress_config = self.compress_config.copy()
        new.peak_bytes = 0
        new.use_dummy_qn = self.use_dummy_qn
        new.qn = [qn.copy() for qn in self.qn]
        new.qnidx = self.qnidx
        new._qntot = self.qntot
        return new

    def array2mt(self, array, idx):
        if isinstance(array, Matrix):
            mt = array
        else:
            mt = Matrix.interned(array, self.is_mpo, dtype=self.dtype)
        if self.use_dummy_qn:
            mt.sigmaqn = np.zeros(mt.pdim_prod, dtype=np.int)
        else:
            mt.sigmaqn = self._get_sigmaqn(idx)
        return mt

    @property
    def total_bytes(self):
        return sum([array.nbytes for array in self])

    def set_peak_bytes(self, new_bytes=None):
        if new_bytes is None:
            new_bytes = self.total_bytes
        if new_bytes < self.peak_bytes:
            return
        self.peak_bytes = new_bytes
        stack = "".join(traceback.format_stack(inspect.stack()[2].frame, 1)).replace(
            "\n", " "
        )
        logger.debug(
            "Set peak bytes to {}. Called from: {}".format(sizeof_fmt(new_bytes), stack)
        )

    def _get_sigmaqn(self, idx):
        raise NotImplementedError

    def get_sigmaqn(self, idx):
        return self[idx].sigmaqn

    def __eq__(self, other):
        for m1, m2 in zip(self, other):
            if not allclose(m1, m2):
                return False
        return True

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "%s with %d sites" % (self.__class__, len(self))

    def __iter__(self):
        return iter(self._mp)

    def __len__(self):
        return len(self._mp)

    def __getitem__(self, item):
        return self._mp[item]

    def __setitem__(self, key, array):
        new_mt = self.array2mt(array, key)
        self._mp[key] = new_mt

    def append(self, array):
        new_mt = self.array2mt(array, len(self))
        self._mp.append(new_mt)

    def clear(self):
        self._mp.clear()
