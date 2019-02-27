# -*- coding: utf-8 -*-

import importlib.util
import logging

import numpy as np


logger = logging.getLogger(__name__)


if importlib.util.find_spec("cupy"):
    import cupy as cp
    xp = cp
    logger.info("use cupy as backend")
else:
    cp = None
    xp = np
    logger.info("use numpy as backend")


#xp = np

class Backend:

    _init_once_flag = False

    def __new__(cls):
        if cls._init_once_flag:
            raise RuntimeError("Backend should only be initialized once")
        cls._init_once_flag = True
        return super().__new__(cls)

    def __init__(self):
        self.first_mp = False
        #self._real_dtype = xp.float64
        #self._complex_dtype = xp.complex128
        self._real_dtype = xp.float32
        self._complex_dtype = xp.complex64

    def sync(self):
        # only works with one GPU
        if xp == cp:
            cp.cuda.device.Device(0).synchronize()

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