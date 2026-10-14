# -*- coding: utf-8 -*-

"""CuPy backend — delegates to cupy if installed, raises clear error if not."""

import logging
import os

import numpy as np

from renormalizer.backend.abstract import AbstractBackend

logger = logging.getLogger(__name__)

_cupy = None
_cupy_available = False


def _try_import_cupy():
    global _cupy, _cupy_available
    try:
        import cupy as cp
        _cupy = cp
        _cupy_available = True
    except ImportError:
        _cupy_available = False


_try_import_cupy()


class CupyBackend(AbstractBackend):
    name = "cupy"
    ndarray = (np.ndarray,)  # will be extended in __init__ if cupy available
    memory_errors = (MemoryError,)
    opt_einsum_name = "cupy"

    def __init__(self, device=None):
        if not _cupy_available:
            raise ImportError(
                "CuPy is not installed. Install cupy or select another backend."
            )
        super().__init__()
        if device is None:
            self._device = _cupy.cuda.Device()
        elif isinstance(device, str) and device.startswith('cuda:') and device[5:].isdigit():
            self._device = _cupy.cuda.Device(int(device[5:]))
        else:
            raise ValueError(f'unsupported CuPy device {device}')
        if device is not None and self._device.id >= _cupy.cuda.runtime.getDeviceCount():
            raise ValueError(f'CuPy device {device} unavailable')
        self.array_namespace = _cupy
        self.ndarray = _cupy.ndarray
        self.array_types = (np.ndarray, _cupy.ndarray)
        self.device_array_types = (_cupy.ndarray,)
        self.memory_errors = (MemoryError, _cupy.cuda.memory.OutOfMemoryError)

        # Like __getattr__ below, run these namespaces on this adapter's device
        # rather than on whichever device is current.
        self.linalg = _DeviceScoped(_cupy.linalg, self._device)
        self.random = _DeviceScoped(_cupy.random, self._device)

        if os.environ.get("RENO_FP32") is not None:
            self.use_32bits()

    def __getattr__(self, name):
        attribute = getattr(_cupy, name)
        if not callable(attribute) or isinstance(attribute, type):
            return attribute
        # CuPy launches work on the thread's current device. Each call enters this
        # adapter's device so that several adapters on different GPUs stay correct;
        # with a single GPU (the legacy RENO_GPU use) this is the same device anyway.
        def invoke(*args, **kwargs):
            with self._device:
                return attribute(*args, **kwargs)
        return invoke

    def krylov_combine(self, V, coefficients):
        # Kept as in the original CuPy code (NumPy-style promotion). Measured on
        # GPU: identical in float64; in float32 the promoted combination is closer
        # to the float64 result than casting the coefficients to the basis dtype.
        with self._device:
            return V @ _cupy.asarray(coefficients)

    def current_device(self):
        return f'cuda:{self._device.id}'

    def device_scope(self):
        return self._device

    def is_array(self, x):
        return isinstance(x, self.array_types)

    def owns(self, x):
        return isinstance(x, _cupy.ndarray) and x.device.id == self._device.id

    def dtype_of(self, x):
        return np.dtype(x.dtype)

    def strict_call(self, name, *args, **kwargs):
        if name == 'astype':
            with self._device:
                return args[0].astype(args[1], **kwargs)
        namespace = _cupy
        for part in name.split('.'):
            namespace = getattr(namespace, part)
        import cupyx
        with self._device, cupyx.errstate(linalg='raise'):
            return namespace(*args, **kwargs)

    def isolate_rng(self):
        """New explicit contexts own an RNG; legacy adapters keep global seeds."""
        with self._device:
            self.random = _CupyRandom(self, _cupy.random.RandomState(2019))

    def strict_update(self, name, x, idx, value):
        with self._device:
            y = x.copy()
            # Ordered scalar updates implement repeated-index semantics for all
            # four dtypes; no atomic-support or duplicate-assignment assumption.
            if isinstance(idx, np.ndarray):
                values = _cupy.broadcast_to(value, idx.shape)
                for position, index in enumerate(idx):
                    if name == 'set':
                        y[int(index)] = values[position]
                    elif name == 'add':
                        y[int(index)] += values[position]
                    elif name == 'sub':
                        y[int(index)] -= values[position]
                    else:
                        y[int(index)] *= values[position]
            elif name == 'set':
                y[idx] = value
            elif name == 'add':
                y[idx] += value
            elif name == 'sub':
                y[idx] -= value
            else:
                y[idx] *= value
            return y

    def array(self, data, dtype=None, *, copy=True, **kwargs):
        if copy is not None and type(copy) is not bool:
            raise TypeError('copy must be None, True, or False')
        if copy is False:
            if not self.owns(data) or (dtype is not None and np.dtype(dtype) != data.dtype):
                raise ValueError('copy=False cannot transfer or convert a CuPy array')
            return data
        with self._device:
            if copy is True:
                return _cupy.array(data, dtype=dtype, copy=True, **kwargs)
            return _cupy.asarray(data, dtype=dtype, **kwargs)

    def asarray(self, data, dtype=None, **kwargs):
        return self.array(data, dtype=dtype, copy=None, **kwargs)

    def from_numpy(self, x, *, copy=None):
        return self.array(x, dtype=x.dtype, copy=copy)

    def to_numpy(self, x, *, copy=None):
        if copy is False:
            raise ValueError('copy=False cannot transfer CuPy data to host')
        with self._device:
            return _cupy.asnumpy(x)

    def numpy(self, x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        return self.to_numpy(x)

    def free_all_blocks(self):
        mempool = _cupy.get_default_memory_pool()
        mempool.free_all_blocks()

    def log_memory_usage(self, header=""):
        from renormalizer.utils.utils import sizeof_fmt
        mempool = _cupy.get_default_memory_pool()
        logger.info(
            f"{header} GPU memory used/Total: "
            f"{sizeof_fmt(mempool.used_bytes())}/{sizeof_fmt(mempool.total_bytes())}"
        )

    def sync(self):
        self._device.synchronize()


class _DeviceScoped:
    """A CuPy namespace whose functions run on a fixed device."""

    def __init__(self, namespace, device):
        self._namespace = namespace
        self._device = device

    def __getattr__(self, name):
        attribute = getattr(self._namespace, name)
        if not callable(attribute) or isinstance(attribute, type):
            return attribute
        device = self._device
        def invoke(*args, **kwargs):
            with device:
                return attribute(*args, **kwargs)
        return invoke


class _CupyRandom:
    def __init__(self, backend, state):
        self._backend = backend
        self._state = state

    def __getattr__(self, name):
        function = getattr(self._state, 'random_sample' if name == 'random' else name)
        def invoke(*args, **kwargs):
            if name in ('random', 'random_sample', 'rand', 'randn', 'normal', 'uniform'):
                if kwargs.get('dtype') is None:
                    kwargs['dtype'] = self._backend.real_dtype
            with self._backend._device:
                return function(*args, **kwargs)
        return invoke
