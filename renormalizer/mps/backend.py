# -*- coding: utf-8 -*-

import os
import logging
import random
import subprocess

import numpy as np

from renormalizer.utils.utils import sizeof_fmt

try:
    import primme
    IMPORT_PRIMME_EXCEPTION = None
except Exception as e:
    primme = None
    IMPORT_PRIMME_EXCEPTION = e


logger = logging.getLogger(__name__)


GPU_KEY = "RENO_GPU"
USE_GPU = False

GPU_ID = os.environ.get(GPU_KEY, None)


def try_import_cupy():
    global GPU_ID

    try:
        import cupy as cp
    except ImportError as e:
        if GPU_ID is not None:
            logger.warning(f"CuPy is not installed. Setting {GPU_KEY} to {GPU_ID} has no effect.")
            logger.exception(e)
        return False, np

    if GPU_ID is None:
        GPU_ID = 0

    try:
        cp.cuda.Device(GPU_ID).use()
    except cp.cuda.runtime.CUDARuntimeError as e:
        logger.warning("Failed to initialize CuPy.")
        logger.exception(e)
        return False, np

    logger.info(f"Using GPU: {GPU_ID}")
    return True, cp

def get_git_commit_hash():
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.PIPE)
        return commit_hash.strip().decode('utf-8')
    except subprocess.CalledProcessError:
        return "Unknown"


USE_GPU, xp = try_import_cupy()


# USE_GPU = False
# xp = np

xpseed = 2019
npseed = 9012
randomseed = 1092

xp.random.seed(xpseed)
np.random.seed(npseed)
random.seed(randomseed)


if not USE_GPU:
    logger.info("Use NumPy as backend")
    logger.info(f"numpy random seed is {npseed}")
    OE_BACKEND = "numpy"
else:
    logger.info("Use CuPy as backend")
    logger.info(f"cupy random seed is {xpseed}")
    OE_BACKEND = "cupy"
logger.info(f"random seed is {randomseed}")
logger.info("Git Commit Hash: %s", get_git_commit_hash())


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
        if os.environ.get("RENO_FP32") is None:
            self.use_64bits()
        else:
            self.use_32bits()

    def free_all_blocks(self):
        if not USE_GPU:
            return
        # free memory
        mempool = xp.get_default_memory_pool()
        mempool.free_all_blocks()

    def log_memory_usage(self, header=""):
        if not USE_GPU:
            return
        mempool = xp.get_default_memory_pool()
        logger.info(f"{header} GPU memory used/Total: {sizeof_fmt(mempool.used_bytes())}/{sizeof_fmt(mempool.total_bytes())}")

    def sync(self):
        # only works with one GPU
        if USE_GPU:
            xp.cuda.device.Device(GPU_ID).synchronize()

    def use_32bits(self):
        logger.info("use 32 bits")
        self.dtypes = (np.float32, np.complex64)

    def use_64bits(self):
        logger.info("use 64 bits")
        self.dtypes = (np.float64, np.complex128)

    @property
    def is_32bits(self) -> bool:
        return self._real_dtype == np.float32

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

    @property
    def canonical_atol(self):
        '''
        Absolute tolerence for use in matrix.check_lortho, 
        mp.check_left_canonical, mp.ensure_left_canonical
        and their right counterparts
        '''
        return (
            self._canonical_atol
            if hasattr(self, "_canonical_atol")
            else (1e-4 if self.is_32bits else 1e-8)
        )

    @property
    def canonical_rtol(self):
        '''
        Relative tolerence for use in matrix.check_lortho, 
        mp.check_left_canonical, mp.ensure_left_canonical
        and their right counterparts
        '''
        return (
            self._canonical_rtol
            if hasattr(self, "_canonical_rtol")
            else (1e-2 if self.is_32bits else 1e-5)
        )

    @canonical_atol.setter
    def canonical_atol(self, value):
        self._canonical_atol = self._tol_checker(value)

    @canonical_rtol.setter
    def canonical_rtol(self, value):
        self._canonical_rtol = self._tol_checker(value)

    def _tol_checker(self, value):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("Tolerance must be a non-negative float number")
        return value


backend = Backend()
