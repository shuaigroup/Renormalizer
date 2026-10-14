# -*- coding: utf-8 -*-

import logging
import os

import numpy as np

from renormalizer.backend.abstract import AbstractBackend
from renormalizer.backend import numpy_contraction

logger = logging.getLogger(__name__)


class NumpyBackend(AbstractBackend):
    name = "numpy"
    array_namespace = np
    ndarray = np.ndarray
    memory_errors = (MemoryError,)
    opt_einsum_name = "numpy"

    @property
    def opt_einsum_module(self):
        # Renormalizer's contractions use the cached tensordot (numpy_contraction).
        return numpy_contraction.__name__

    def __init__(self):
        super().__init__()
        if os.environ.get("RENO_FP32") is not None:
            self.use_32bits()

        self.linalg = np.linalg
        self.random = np.random

    def __getattr__(self, name):
        return getattr(np, name)

    # Bitwise-identical to numpy.tensordot, with its bookkeeping cached by shape.
    tensordot = staticmethod(numpy_contraction.tensordot)

    def array(self, data, dtype=None, *, copy=True, **kwargs):
        """Construct an array; copy=False is a strict no-allocation promise.

        Negative strides, read-only and overlapping views are preserved when no
        copy is requested. Empty views preserve provenance even though no bytes
        overlap. Legacy unspecified dtype inference is intentionally unchanged.
        """
        if copy is not None and type(copy) is not bool:
            raise TypeError("copy must be None, True, or False")
        if copy is False:
            if not isinstance(data, np.ndarray):
                raise ValueError("copy=False requires an existing NumPy array")
            if dtype is not None and data.dtype != np.dtype(dtype):
                raise ValueError("copy=False cannot change dtype")
            order = kwargs.pop("order", "K")
            ndmin = kwargs.pop("ndmin", 0)
            subok = kwargs.pop("subok", False)
            if kwargs:
                raise TypeError(f"unsupported options: {sorted(kwargs)}")
            if order not in ("K", "A", "C", "F"):
                raise ValueError("invalid order")
            if order == "A" and not (data.flags.c_contiguous or data.flags.f_contiguous):
                raise ValueError("copy=False cannot satisfy contiguous A order")
            if order == "C" and not data.flags.c_contiguous:
                raise ValueError("copy=False cannot satisfy C order")
            if order == "F" and not data.flags.f_contiguous:
                raise ValueError("copy=False cannot satisfy F order")
            if not isinstance(ndmin, int) or ndmin < 0:
                raise ValueError("ndmin must be a non-negative integer")
            out = data if subok else np.asarray(data)
            if ndmin > out.ndim:
                out = out.reshape((1,) * (ndmin - out.ndim) + out.shape)
            return out
        if copy is True:
            return np.array(data, dtype=dtype, copy=True, **kwargs)
        if not kwargs:
            return np.asarray(data, dtype=dtype)
        return np.array(data, dtype=dtype, **kwargs)

    def asarray(self, *args, **kwargs):
        return np.asarray(*args, **kwargs)

    def from_numpy(self, x, *, copy=None):
        return self.array(x, copy=copy)

    def to_numpy(self, x, *, copy=None):
        return self.array(x, copy=copy)

    def numpy(self, x):
        if x is None:
            return None
        return self.to_numpy(x)
