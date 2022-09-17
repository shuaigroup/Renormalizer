# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import weakref
import logging
from typing import List, Union

from renormalizer.mps.backend import np, backend, xp, USE_GPU

logger = logging.getLogger(__name__)


class Matrix:

    def __init__(self, array, dtype=None):
        assert array is not None
        array = asnumpy(array)
        if dtype == backend.real_dtype:
            # forbid unchecked casting
            assert not np.iscomplexobj(array)
        if dtype is None:
            if np.iscomplexobj(array):
                dtype = backend.complex_dtype
            else:
                dtype = backend.real_dtype
        self.array: np.ndarray = np.asarray(array, dtype=dtype)
        self.original_shape = self.array.shape
        self.sigmaqn = None
        backend.running = True

    def __getattr__(self, item):
        # use this way to obtain ``array`` to prevent infinite recursion during multi-processing
        # see https://stackoverflow.com/questions/22781872/python-pickle-got-acycle-recursion-with-getattr
        array = super().__getattribute__("array")
        res = getattr(array, item)
        if isinstance(res, np.ndarray):
            return Matrix(res)
        functiontype = type([].append)
        if isinstance(res, functiontype):

            def wrapped(*args, **kwargs):
                res2 = res(*args, **kwargs)
                if isinstance(res2, np.ndarray):
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
        self.array = np.asarray(self.array, dtype=dtype)
        return self

    def abs(self):
        return self.__class__(np.abs(self.array))

    def norm(self):
        return np.linalg.norm(self.array.flatten())

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

    def check_lortho(self, atol=None):
        """
        check L-orthogonal
        """
        if atol is None:
            atol = backend.canonical_atol
        tensm = asxp(self.array.reshape([np.prod(self.shape[:-1]), self.shape[-1]]))
        s = tensm.T.conj() @ tensm
        return xp.allclose(s, xp.eye(s.shape[0]), atol=atol)

    def check_rortho(self, atol=None):
        """
        check R-orthogonal
        """
        if atol is None:
            atol = backend.canonical_atol
        tensm = asxp(self.array.reshape([self.shape[0], np.prod(self.shape[1:])]))
        s = tensm @ tensm.T.conj()
        return xp.allclose(s, xp.eye(s.shape[0]), atol=atol)

    def to_complex(self):
        # `xp.array` always creates new array, so to_complex means copy, which is
        # in accordance with NumPy
        return np.array(self.array, dtype=backend.complex_dtype)

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
        return np.allclose(self.array, np.zeros_like(self.array), atol=atol)

    def __hash__(self):
        return hash((self.array.shape, self.array.tobytes()))

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

    def __complex__(self):
        return self.array.__complex__()


def zeros(shape, dtype=None):
    if dtype is None:
        dtype = backend.real_dtype
    return Matrix(np.zeros(shape), dtype=dtype)


def eye(N, M=None, dtype=None):
    if dtype is None:
        dtype = backend.real_dtype
    return Matrix(np.eye(N, M), dtype=dtype)


def ones(shape, dtype=None):
    if dtype is None:
        dtype = backend.real_dtype
    return Matrix(np.ones(shape), dtype=dtype)


def einsum(subscripts, *operands):
    return Matrix(np.einsum(subscripts, *[o.array for o in operands]))


def tensordot(a: Union[Matrix, np.ndarray], b: Union[Matrix, np.ndarray, xp.ndarray], axes) -> xp.ndarray:
    return xp.tensordot(asxp(a), asxp(b), axes)


def moveaxis(a: Matrix, source, destination):
    return Matrix(np.moveaxis(a.array, source, destination))


def vstack(tup):
    return Matrix(np.vstack([m.array for m in tup]))


def dstack(tup):
    return Matrix(np.dstack([m.array for m in tup]))


def concatenate(arrays, axis=None):
    return Matrix(np.concatenate([m.array for m in arrays], axis))


# can only use numpy for now. see gh-cupy-1946
def allclose(a, b, rtol=1.0e-5, atol=1.0e-8):
    if isinstance(a, Matrix):
        a = a.array
    else:
        a = np.asarray(a)
    if isinstance(b, Matrix):
        b = b.array
    else:
        b = np.asarray(b)
    return np.allclose(a, b, rtol=rtol, atol=atol)


def multi_tensor_contract(path, *operands: [List[Union[Matrix, np.ndarray, xp.ndarray]]]):
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
    view_left: Union[Matrix, np.ndarray, xp.ndarray],
    input_left,
    view_right: Union[Matrix, np.ndarray, xp.ndarray],
    input_right,
    idx_removed,
):
    # Find indices to contract over
    left_pos, right_pos = (), ()
    for s in idx_removed:
        left_pos += (input_left.find(s),)
        right_pos += (input_right.find(s),)
    return tensordot(view_left, view_right, axes=(left_pos, right_pos))


def asnumpy(array: Union[np.ndarray, xp.ndarray, Matrix]) -> np.ndarray:
    if array is None:
        return None
    if isinstance(array, Matrix):
        return array.array
    if not USE_GPU:
        assert isinstance(array, np.ndarray)
        return array
    if isinstance(array, np.ndarray):
        return array
    stream = xp.cuda.get_current_stream()
    return xp.asnumpy(array, stream=stream)


def asxp(array: Union[np.ndarray, xp.ndarray, Matrix]) -> xp.ndarray:
    if array is None:
        return None
    if isinstance(array, Matrix):
        array = array.array
    if not USE_GPU:
        assert isinstance(array, np.ndarray)
        return array
    return xp.asarray(array)
