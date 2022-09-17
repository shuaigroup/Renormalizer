# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
import os
import sys
import warnings


reno_num_threads = os.environ.get("RENO_NUM_THREADS")
if reno_num_threads is not None:
    # set environment variables to limit NumPy cpu usage
    # Note that this should be done before NumPy is imported

    if "numpy" in sys.modules:
        warnings.warn("renormalizer should be imported before numpy for `RENO_NUM_THREADS` to take effect")
    else:
        for env in ["MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"]:
            os.environ[env] = reno_num_threads
            del env

del os, sys, warnings

from renormalizer.utils.log import init_log

init_log()

del init_log
