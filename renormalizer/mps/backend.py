# -*- coding: utf-8 -*-

import os
import importlib.util
import logging
import random

import numpy as np

from renormalizer.utils.utils import sizeof_fmt


logger = logging.getLogger(__name__)


GPU_KEY = "EPHMPS_GPU"


if importlib.util.find_spec("cupy"):
    import cupy as cp
    xp = cp
    gpu_id = os.environ.get(GPU_KEY, 0)
    logger.info(f"Using GPU: {gpu_id}")
    cp.cuda.Device(gpu_id).use()
else:
    gpu_id = os.environ.get(GPU_KEY, None)
    if gpu_id is not None:
        logger.warning(f"Cupy is not installed. Setting {GPU_KEY} to {gpu_id} has no effect.")
    cp = None
    xp = np



# xp = np

if xp is np:
    logger.info("use numpy as backend")
elif xp is cp:
    logger.info("use cupy as backend")
else:
    assert False


np.random.seed(9012)
xp.random.seed(2019)
random.seed(1092)


class Backend:

    _init_once_flag = False

    def __new__(cls):
        if cls._init_once_flag:
            raise RuntimeError("Backend should only be initialized once")
        cls._init_once_flag = True
        return super().__new__(cls)

    def __init__(self):
        self.first_mp = False
        self._real_dtype = None
        self._complex_dtype = None
        #self.use_32bits()
        self.use_64bits()

    def free_all_blocks(self):
        if xp == np:
            return
        # free memory
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()

    def log_memory_usage(self, header=""):
        if xp == np:
            return
        mempool = cp.get_default_memory_pool()
        logger.info(f"{header} GPU memory used/Total: {sizeof_fmt(mempool.used_bytes())}/{sizeof_fmt(mempool.total_bytes())}")

    def sync(self):
        # only works with one GPU
        if xp == cp:
            cp.cuda.device.Device(0).synchronize()

    def use_32bits(self):
        logger.info("use 32 bits")
        self.dtypes = (xp.float32, xp.complex64)

    def use_64bits(self):
        logger.info("use 64 bits")
        self.dtypes = (xp.float64, xp.complex128)

    @property
    def is_32bits(self) -> bool:
        return self._real_dtype == xp.float32

    @property
    def real_dtype(self):
        return self._real_dtype

    @real_dtype.setter
    def real_dtype(self, tp):
        if not self.first_mp:
            self._real_dtype = tp
        else:
            raise RuntimeError("Can't alter backend data type")

    @property
    def complex_dtype(self):
        return self._complex_dtype

    @complex_dtype.setter
    def complex_dtype(self, tp):
        if not self.first_mp:
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
