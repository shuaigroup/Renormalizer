# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import weakref
import logging
from typing import List, Union

import numpy as np

from renormalizer.mps.backend import xp, backend, cp


logger = logging.getLogger(__name__)


class Matrix:

    _mo_dict = weakref.WeakValueDictionary()  # dummy value

    @classmethod
    def interned(cls, array, is_mpo, dtype):
        new_matrix = cls(array, dtype)
        if is_mpo:
            h = hash(new_matrix)
            old_matrix = cls._mo_dict.get(h, None)
            if old_matrix is not None:
                assert allclose(old_matrix.array, new_matrix.array)
                return old_matrix
            cls._mo_dict[h] = new_matrix
        return new_matrix

    def __init__(self, array, dtype=None, is_full_mpdm=False):
        assert array is not None
        if dtype == backend.real_dtype:
            # forbid unchecked casting
            assert not xp.iscomplexobj(array)
        if dtype is None:
            if xp.iscomplexobj(array):
                dtype = backend.complex_dtype
            else:
                dtype = backend.real_dtype
        self.array: xp.ndarray = xp.asarray(array, dtype=dtype)
        self.original_shape = self.array.shape
        self.sigmaqn = None
        self.is_full_mpdm = is_full_mpdm
        backend.running = True

    def __getattr__(self, item):
        # use this way to obtain ``array`` to prevent infinite recursion during multi-processing
        # see https://stackoverflow.com/questions/22781872/python-pickle-got-acycle-recursion-with-getattr
        array = super().__getattribute__("array")
        res = getattr(array, item)
        if isinstance(res, xp.ndarray):
            return Matrix(res)
        functiontype = type([].append)
        if isinstance(res, functiontype):

            def wrapped(*args, **kwargs):
                res2 = res(*args, **kwargs)
                if isinstance(res2, xp.ndarray):
                    return Matrix(res2)
                return res2

            return wrapped
        return res

    # for debugging purpose (let it shown in debuggers)
    @property
    def dtype(self):
        return self.array.dtype

    def astype(self, dtype):
        assert not (self.dtype == backend.complex_dtype and dtype == backend.real_dtype)
        self.array = xp.asarray(self.array, dtype=dtype)
        return self

    def abs(self):
        return self.__class__(xp.abs(self.array))

    def norm(self):
        return xp.linalg.norm(self.array.flatten())

    def asnumpy(self):
        if xp == np:
            return self.array
        else:
            return xp.asnumpy(self.array)

    # physical indices exclude first and last indices
    @property
    def pdim(self):
        return self.original_shape[1:-1]

    @property
    def pdim_prod(self):
        return np.prod(self.pdim)

    @property
    def bond_dim(self):
        return self.original_shape[0], self.original_shape[-1]

    @property
    def r_combine_shape(self):
        return self.original_shape[0], np.prod(self.original_shape[1:])

    @property
    def l_combine_shape(self):
        return np.prod(self.original_shape[:-1]), self.original_shape[-1]

    def r_combine(self):
        return self.reshape(self.r_combine_shape)

    def l_combine(self):
        return self.reshape(self.l_combine_shape)

    def check_lortho(self, atol=1e-8):
        """
        check L-orthogonal
        """
        tensm = self.array.reshape([np.prod(self.shape[:-1]), self.shape[-1]])
        s = xp.dot(tensm.T.conj(), tensm)
        return allclose(s, xp.eye(s.shape[0]), atol=atol)

    def check_rortho(self, atol=1e-8):
        """
        check R-orthogonal
        """
        tensm = self.array.reshape([self.shape[0], np.prod(self.shape[1:])])
        s = xp.dot(tensm, tensm.T.conj())
        return allclose(s, xp.eye(s.shape[0]), atol=atol)

    def to_complex(self, inplace=False):
        # `xp.array` always creates new array, so to_complex means copy, which is
        # in accordance with NumPy
        if inplace:
            self.array = xp.array(self.array, dtype=backend.complex_dtype)
            return self
        else:
            return xp.array(self.array, dtype=backend.complex_dtype)

    def copy(self):
        new = self.__class__(self.array.copy(), self.array.dtype)
        new.original_shape = self.original_shape
        new.sigmaqn = self.sigmaqn
        return new

    def nearly_zero(self):
        if backend.is_32bits:
            atol = 1e-10
        else:
            atol = 1e-20
        return xp.allclose(self.array, xp.zeros_like(self.array), atol=atol)

    def __hash__(self):
        if xp == np:
            return hash(self.array.tobytes())
        else:
            # the transfer shouldn't be a bottleneck as they are all
            # small matrices. Could be optimized to hash on device
            return hash(xp.asnumpy(self.array).tobytes())

    def __getitem__(self, item):
        res = self.array.__getitem__(item)
        if res.ndim != 0:
            return self.__class__(res)
        else:
            return res

    def __setitem__(self, key, value):
        if isinstance(value, Matrix):
            value = value.array
        self.array[key] = value

    def __add__(self, other):
        if isinstance(other, Matrix):
            other = other.array
        return self.__class__(self.array.__add__(other))

    def __radd__(self, other):
        if isinstance(other, Matrix):
            other = other.array
        return self.__class__(self.array.__radd__(other))

    def __mul__(self, other):
        if isinstance(other, Matrix):
            other = other.array
        return self.__class__(self.array.__mul__(other))

    def __rmul__(self, other):
        if isinstance(other, Matrix):
            other = other.array
        return self.__class__(self.array.__rmul__(other))

    def __truediv__(self, other):
        if isinstance(other, Matrix):
            other = other.array
        return self.__class__(self.array.__truediv__(other))

    def __repr__(self):
        return f"<Matrix at 0x{id(self):x} {self.shape} {self.dtype}>"

    def __str__(self):
        return str(self.array)

    def __float__(self):
        return self.array.__float__()


def zeros(shape, dtype=None):
    if dtype is None:
        dtype = backend.real_dtype
    return Matrix(xp.zeros(shape), dtype=dtype)


def eye(N, M=None, dtype=None):
    if dtype is None:
        dtype = backend.real_dtype
    return Matrix(xp.eye(N, M), dtype=dtype)


def ones(shape, dtype=None):
    if dtype is None:
        dtype = backend.real_dtype
    return Matrix(xp.ones(shape), dtype=dtype)


def einsum(subscripts, *operands):
    return Matrix(xp.einsum(subscripts, *[o.array for o in operands]))


def tensordot(a: Union[Matrix, xp.ndarray], b: Union[Matrix, xp.ndarray], axes):
    matrix_flag = False
    if isinstance(a, Matrix):
        a = a.array
        assert isinstance(b, Matrix)
        b = b.array
        matrix_flag = True
    else:
        assert isinstance(b, xp.ndarray)
    res = xp.tensordot(a, b, axes)
    if matrix_flag:
        return Matrix(res)
    else:
        return res


def moveaxis(a: Matrix, source, destination):
    return Matrix(xp.moveaxis(a.array, source, destination))


def vstack(tup):
    return Matrix(xp.vstack([m.array for m in tup]))


def dstack(tup):
    return Matrix(xp.dstack([m.array for m in tup]))


def concatenate(arrays, axis=None):
    return Matrix(xp.concatenate([m.array for m in arrays], axis))


# can only use numpy for now. see gh-cupy-1946
def allclose(a, b, rtol=1.0e-5, atol=1.0e-8):
    if isinstance(a, Matrix):
        a = a.array
    else:
        a = xp.asarray(a)
    if isinstance(b, Matrix):
        b = b.array
    else:
        b = xp.asarray(b)
    # delete this when CuPy 6.0 is released
    if xp == cp:
        a = cp.asnumpy(a)
        b = cp.asnumpy(b)
    return np.allclose(a, b, rtol=rtol, atol=atol)


def asnumpy(a: xp.ndarray):
    if xp == np:
        return a
    else:
        return xp.asnumpy(a)


def multi_tensor_contract(path, *operands: [List[Union[Matrix, xp.ndarray]]]):
    """
    ipath[0] is the index of the mat
    ipaht[1] is the contraction index
    oeprands is the arrays

    For example:  in mpompsmat.py
    path = [([0, 1],"fdla, abc -> fdlbc")   ,\
            ([2, 0],"fdlbc, gdeb -> flcge") ,\
            ([1, 0],"flcge, helc -> fgh")]
    outtensor = tensorlib.multi_tensor_contract(path, MPSconj[isite], intensor,
            MPO[isite], MPS[isite])
    """

    operands = list(operands)
    for ipath in path:

        input_str, results_str = ipath[1].split("->")
        input_str = input_str.split(",")
        input_str = [x.replace(" ", "") for x in input_str]
        results_set = set(results_str)
        inputs_set = set(input_str[0] + input_str[1])
        idx_removed = inputs_set - (inputs_set & results_set)

        tmpmat = pair_tensor_contract(
            operands[ipath[0][0]],
            input_str[0],
            operands[ipath[0][1]],
            input_str[1],
            idx_removed,
        )

        for x in sorted(ipath[0], reverse=True):
            del operands[x]

        operands.append(tmpmat)

    return operands[0]


def pair_tensor_contract(
    view_left: Union[Matrix, xp.ndarray],
    input_left,
    view_right: Union[Matrix, xp.ndarray],
    input_right,
    idx_removed,
):
    # Find indices to contract over
    left_pos, right_pos = (), ()
    for s in idx_removed:
        left_pos += (input_left.find(s),)
        right_pos += (input_right.find(s),)
    return tensordot(view_left, view_right, axes=(left_pos, right_pos))


class EmptyMatrixError(Exception): pass
