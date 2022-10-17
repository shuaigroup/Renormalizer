# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
import logging
import os
import sys
import warnings


reno_num_threads = os.environ.get("RENO_NUM_THREADS")
if reno_num_threads is not None:
    # Set environment variables to limit NumPy CPU usage
    # Note that this should be done before NumPy is imported

    if "numpy" in sys.modules:
        warnings.warn("renormalizer should be imported before numpy for `RENO_NUM_THREADS` to take effect")
    else:
        for env in ["MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"]:
            os.environ[env] = reno_num_threads
            del env

del sys, warnings

from renormalizer.utils.log import init_log


# logging.DEBUG == 10, logging.INFO == 20. See the logging built-in package for more
log_level = int(os.environ.get("RENO_LOG_LEVEL", logging.DEBUG))

init_log(log_level)

del os, log_level, init_log


# user interfaces
from renormalizer.model import Model, HolsteinModel, SpinBosonModel, TI1DModel, Op
from renormalizer.model.basis import BasisSHO, BasisHopsBoson, BasisSineDVR, BasisMultiElectron, \
    BasisMultiElectronVac, BasisSimpleElectron, BasisHalfSpin
from renormalizer.mps import Mpo, Mps, optimize_mps
from renormalizer.utils import Quantity
