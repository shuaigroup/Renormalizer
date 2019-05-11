# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
from __future__ import absolute_import, division

import inspect
import logging
import traceback
from typing import List, Union

import numpy as np

from ephMPS.model import MolList, EphTable
from ephMPS.mps import svd_qn
from ephMPS.mps.matrix import (
    einsum,
    eye,
    allclose,
    backend,
    vstack,
    dstack,
    concatenate,
    zeros,
    tensordot,
    Matrix,
    EmptyMatrixError)
from ephMPS.utils import sizeof_fmt, CompressConfig

logger = logging.getLogger(__name__)


class MatrixProduct:

    @classmethod
    def load(cls, mol_list: MolList, fname: str):
        npload = np.load(fname)
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
        mp.left = bool(npload["left"])
        return mp

    def __init__(self):
        # XXX: when modify theses codes, keep in mind to update `metacopy` method
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
        # sweeping from left?
        self.left: bool = None

        # compress after add?
        self.compress_add: bool = False

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
    def bond_dims(self) -> List:
        bond_dims = (
            [mt.bond_dim[0] for mt in self] + [self[-1].bond_dim[-1]]
            if self.site_num
            else []
        )
        # return a list so that the logging result is more pretty
        return bond_dims

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
        if self._pbond_list is None:
            self._pbond_list = self.mol_list.pbond_list
        return self._pbond_list

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
        # retain qnidx to indicate left or right canonicalise
        if self.qnidx is None:
            self.qnidx = 0
        self.qn = [[0] * dim for dim in self.bond_dims]
        if self.left is None:
            self.left = True

    def build_none_qn(self):
        self.qntot = None
        self.qnidx = None
        self.qn = None
        self.left = None

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
        assert self.qnidx in (self.site_num - 1, 0)
        return self.qnidx == self.site_num - 1

    @property
    def is_right_canon(self):
        assert self.qnidx in (self.site_num - 1, 0)
        return self.qnidx == 0

    def iter_idx_list(self, full: bool):
        # if not `full`, the last site is omitted.
        if self.is_left_canon:
            last = -1 if full else 0
            return range(self.site_num - 1, last, -1)
        elif self.is_right_canon:
            last = self.site_num if full else self.site_num - 1
            return range(0, last)
        else:
            # this could happen when canonicalization is break at some point
            last = self.site_num if full else self.site_num - 1
            return range(self.qnidx, last)

    def _update_ms(
        self, idx: int, u: Matrix, vt: Matrix, sigma=None, qnlset=None, qnrset=None, m_trunc=None
    ):
        if m_trunc is None:
            m_trunc = u.shape[1]
        u = u[:, :m_trunc]
        vt = vt[:m_trunc, :]
        if sigma is None:
            # canonicalise, vt is not unitary
            if self.is_mpo and self.left:
                norm = vt.norm()
                u = Matrix(u.array * norm)
                vt = Matrix(vt.array / norm)
        else:
            sigma = sigma[:m_trunc]
            if (not self.is_mpo and self.left) or (self.is_mpo and not self.left):
                vt = einsum("i, ij -> ij", sigma, vt)
            else:
                u = einsum("ji, i -> ji", u, sigma)
        if self.left:
            self[idx + 1] = tensordot(vt, self[idx + 1], axes=1)
            ret_mpsi = u.reshape(
                [u.shape[0] // self[idx].pdim_prod] + list(self[idx].pdim) + [m_trunc]
            )
            if qnlset is not None:
                self.qn[idx + 1] = qnlset[:m_trunc]
        else:
            self[idx - 1] = tensordot(self[idx - 1], u, axes=1)
            ret_mpsi = vt.reshape(
                [m_trunc] + list(self[idx].pdim) + [vt.shape[1] // self[idx].pdim_prod]
            )
            if qnrset is not None:
                self.qn[idx] = qnrset[:m_trunc]
        if ret_mpsi.nbytes < ret_mpsi.base.nbytes * 0.8:
            # do copy here to discard unnecessary data. Note that in NumPy common slicing returns
            # a `view` containing the original data. If `ret_mpsi` is used directly the original
            # `u` or `vt` is not garbage collected.
            ret_mpsi = ret_mpsi.copy()
        assert ret_mpsi.any()
        self[idx] = ret_mpsi

    def _switch_direction(self):
        assert self.left is not None
        if self.left:
            self.qnidx = self.site_num - 1
            self.left = False
            # assert self.check_left_canonical()
        else:
            self.qnidx = 0
            self.left = True
            # assert self.check_right_canonical()


    def _get_big_qn(self, idx):
        mt: Matrix = self[idx]
        sigmaqn = mt.sigmaqn
        qnl = np.array(self.qn[idx])
        qnr = np.array(self.qn[idx + 1])
        assert len(qnl) == mt.shape[0]
        assert len(qnr) == mt.shape[-1]
        assert len(sigmaqn) == mt.pdim_prod
        if self.left:
            qnbigl = np.add.outer(qnl, sigmaqn)
            qnbigr = qnr
        else:
            qnbigl = qnl
            qnbigr = np.add.outer(sigmaqn, qnr)
        return qnbigl, qnbigr

    def add(self, other):
        assert self.qntot == other.qntot
        assert self.site_num == other.site_num
        assert self.qnidx == other.qnidx

        new_mps = other.metacopy()
        if self.is_complex:
            new_mps.to_complex(inplace=True)
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
                    dtype=new_mps.dtype,
                )
                new_ms[: mta.shape[0], :, : mta.shape[2]] = mta
                new_ms[mta.shape[0] :, :, mta.shape[2] :] = mtb
                new_mps[i] = new_ms

            new_mps[-1] = vstack([self[-1], other[-1]])
        elif self.is_mpo or self.is_mpdm:  # MPO
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
                    dtype=new_mps.dtype,
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

    def compress(self):
        """
        inp: canonicalise MPS (or MPO)

        side='l': compress LEFT-canonicalised MPS
                  by sweeping from RIGHT to LEFT
                  output MPS is right canonicalised i.e. CRRR

        side='r': reverse of 'l'

        returns:
             truncated MPS
        """
        if not self.is_mpo:
            # ensure mps is canonicalised. This is time consuming.
            # to disable this, run python as `python -O`
            if self.is_left_canon:
                assert self.check_left_canonical()
            else:
                assert self.check_right_canonical()
        system = "L" if self.left else "R"

        for idx in self.iter_idx_list(full=False):
            mt: Matrix = self[idx]
            if mt.nearly_zero():
                raise EmptyMatrixError
            if self.left:
                mt = mt.l_combine()
            else:
                mt = mt.r_combine()
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
                sigma, idx, self.left
            )
            self._update_ms(
                idx, Matrix(u), Matrix(vt), Matrix(sigma), qnlset, qnrset, m_trunc
            )

        self._switch_direction()
        return self

    def canonicalise(self, stop_idx: int=None):
        for idx in self.iter_idx_list(full=False):
            self.qnidx = idx
            if stop_idx is not None and idx == stop_idx:
                break
            mt: Matrix = self[idx]
            assert mt.any()
            if self.left:
                mt = mt.l_combine()
            else:
                mt = mt.r_combine()
            qnbigl, qnbigr = self._get_big_qn(idx)
            system = "L" if self.left else "R"
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
        self._switch_direction()
        return self

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
        # for debugging. It has little computational cost anyway
        debug_t = []
        for mt1, mt2 in zip(self, other):
            # sum_x e0[:,x].m[x,:,:]
            debug_t.append(e0)
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
        # np.iscomplex regards 1+0j as non complex while np.iscomplexobj
        # regards 1+0j as complex. The former is the desired behavior
        if np.iscomplex(val):
            new_mp.to_complex(inplace=True)
            # no need to care about negative because no `abs` will be done
            negative = False
        else:
            if isinstance(val, complex):
                val = val.real
            negative = val < 0
            val = abs(val)
        # Note matrices are read-only
        # there are two ways to do the scaling
        if np.abs(np.log(np.abs(val))) < 0.01:
            # Thr first way. The operation performs very quickly,
            # but leads to high float point error when val is very large or small
            # val = 2 is considered as very large because the normalization can
            # be done successively
            assert new_mp[self.qnidx].array.any()
            new_mp[self.qnidx] = new_mp[self.qnidx] * val
        else:
            # The second way. High time complexity but numerically more feasible.
            # take care of emtpy matrices. happens at zero electron state at scheme 4
            candidates = list(filter(lambda x: x[1].array.any(), enumerate(self)))
            root_val = val ** (1 / len(candidates))
            for idx, mt in candidates:
                new_mp[idx] = mt * root_val
        if negative:
            new_mp[0] *= -1
        return new_mp

    def to_complex(self, inplace=False):
        if inplace:
            new_mp = self
        else:
            new_mp = self.metacopy()
        new_mp.dtype = backend.complex_dtype
        for i, mt in enumerate(self):
            if mt is None:
                # dummy mt after metacopy. Bad idea. Remove the dummy thing when feasible
                continue
            new_mp[i] = mt.to_complex(inplace)
        return new_mp

    def distance(self, other):
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
        new.left = self.left
        new.compress_add = self.compress_add
        return new

    def array2mt(self, array, idx):
        if isinstance(array, Matrix):
            mt = array.astype(self.dtype)
        else:
            mt = Matrix.interned(array, self.is_mpo, dtype=self.dtype)
        if self.use_dummy_qn:
            mt.sigmaqn = np.zeros(mt.pdim_prod, dtype=np.int)
        else:
            mt.sigmaqn = self._get_sigmaqn(idx)
        # mol_list is None when using quasiboson
        if self.mol_list is None or self.mol_list.scheme != 4:
            if mt.nearly_zero():
                pass
                #raise EmptyMatrixError
        return mt

    @property
    def total_bytes(self):
        return sum(array.nbytes for array in self)

    def set_peak_bytes(self, new_bytes=None):
        if new_bytes is None:
            new_bytes = self.total_bytes
        if new_bytes < self.peak_bytes:
            return
        self.peak_bytes = new_bytes
        want_to_debug_memory = False
        if not want_to_debug_memory:
            return
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

    def set_threshold(self, val):
        self.compress_config.threshold = val

    def dump(self, fname):
        data_dict = dict()
        # version of the protocol
        data_dict["version"] = "0.1"
        data_dict["nsites"] = len(self)
        for idx, mt in enumerate(self):
            data_dict[f"mt_{idx}"] = mt.asnumpy()
        for attr in ["qn", "qnidx", "qntot", "left"]:
            data_dict[attr] = getattr(self, attr)
        try:
            np.savez(fname, **data_dict)
        except Exception as e:
            logger.error(f"Dump mps failed, exception info: f{e}")

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

    def __mul__(self, other):
        assert isinstance(other, (float, complex))
        return self.scale(other)

    def __rmul__(self, other):
        assert isinstance(other, (float, complex))
        return self.scale(other)

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
