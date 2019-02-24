# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import weakref
import logging

import numpy as np

xp = np


logger = logging.getLogger(__name__)


first_mp = False


class Backend:
    def __init__(self):
        self._real_dtype = np.float64
        self._complex_dtype = np.complex128

    @property
    def real_dtype(self):
        return self._real_dtype

    @real_dtype.setter
    def real_dtype(self, tp):
        if not first_mp:
            self._real_dtype = tp
        else:
            raise RuntimeError("Can't alter backend data type")

    @property
    def complex_dtype(self):
        return self._complex_dtype

    @complex_dtype.setter
    def complex_dtype(self, tp):
        if not first_mp:
            self._complex_dtype = tp
        else:
            raise RuntimeError("Can't alter backend data type")


    @property
    def dtypes(self):
        return self.real_dtype, self.complex_dtype

    @dtypes.setter
    def dtypes(self, target):
        self.real_dtype, self.complex_dtype = target


backend = Backend()


'''
class Matrix:

    _mo_dict = weakref.WeakValueDictionary()

    def __new__(cls, array, is_mpo=False):
        if is_mpo:
            k = hash(array.tobytes())
            if k in cls._mo_dict:
                res = cls._mo_dict[k]
                assert np.allclose(res.array, array)
                return res
            else:
                res = super().__new__(cls)
                cls._mo_dict[k] = res
                return res
        else:
            return super().__new__(cls)


    def __init__(self, array, is_mpo=False):
        self.array = array
        self.is_mpo = is_mpo
        self.original_shape = self.array.shape
        self.sigmaqn = None

    @property
    def shape(self):
        return self.array.shape

    @shape.setter
    def shape(self, shape):
        self.array.shape = shape

    @property
    def T(self):
        return self.__class__(self.array.T)

    def conj(self):
        return self.__class__(self.array.conj())

    def reshape(self, shape):
        return self.__class__(self.array.reshape(shape))

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

    def check_lortho(self):
        """
        check L-orthogonal
        """
        tensm = self.reshape([np.prod(self.shape[:-1]), self.shape[-1]])
        s = xp.dot(tensm.T.conj(), tensm)
        return np.allclose(s, np.eye(s.shape[0]), atol=1e-3)

    def check_rortho(self):
        """
        check R-orthogonal
        """
        tensm = self.reshape([self.shape[0], np.prod(self.shape[1:])])
        s = np.dot(tensm, tensm.T.conj())
        return np.allclose(s, np.eye(s.shape[0]))

    def __hash__(self):
        return hash(self.array.tobytes())

    def __deepcopy__(self, memodict):
        return self.__class__(self.array, is_mpo=self.is_mpo)
'''

from functools import wraps

def _mo_check_decorator(func):
    @wraps(func)  # implement the dict by hand because can't override __eq__
    def wrapper(cls, array: np.ndarray, is_mpo: bool, dtype):
        if is_mpo:
            k = hash(array.tobytes())
            if k in cls._mo_dict:
                res = cls._mo_dict[k]
                assert np.allclose(res, array)
                return res
            else:
                res = func(cls, array, is_mpo, dtype)
                cls._mo_dict[k] = res
                return res
        else:
            return func(cls, array, is_mpo, dtype)

    return wrapper

class Matrix(np.ndarray):

    _mo_dict = weakref.WeakValueDictionary()

    @_mo_check_decorator
    def __new__(cls, array, is_mpo: bool, dtype):
        global first_mp
        first_mp = True
        array = np.array(array, dtype=dtype)
        obj = array.view(cls)
        # let hashing possible
        obj.flags.writeable = False
        obj.original_shape = obj.shape
        # set in MatrixProduct
        obj.sigmaqn = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.original_shape = getattr(obj, "original_shape", None)

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

    @property
    def is_mpo(self):
        return self.original_shape.ndim == 4

    def r_combine(self):
        return self.reshape(self.r_combine_shape)

    def l_combine(self):
        return self.reshape(self.l_combine_shape)

    def check_lortho(self):
        """
        check L-orthogonal
        """
        tensm = np.reshape(self, [np.prod(self.shape[:-1]), self.shape[-1]])
        s = np.dot(np.conj(tensm.T), tensm)
        return np.allclose(s, np.eye(s.shape[0]), atol=1e-3)

    def check_rortho(self):
        """
        check R-orthogonal
        """
        tensm = np.reshape(self, [self.shape[0], np.prod(self.shape[1:])])
        s = np.dot(tensm, np.conj(tensm.T))
        return np.allclose(s, np.eye(s.shape[0]))

    def to_complex(self):
        return np.array(self, dtype=backend.complex_dtype)
    

    def __deepcopy__(self, memodict, *arg, **kwargs):
        y = super(Matrix, self).__deepcopy__(memodict, *arg, **kwargs)
        y.original_shape = self.original_shape
        y.sigmaqn = self.sigmaqn
        return y

    def __hash__(self):
        return hash(self.tobytes())
