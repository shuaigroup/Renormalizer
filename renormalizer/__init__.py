# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
import os
import sys

if "numpy" in sys.modules:
    raise ImportError("renormalizer should be imported before numpy is imported")

# set environment variables to limit NumPy cpu usage
# Note that this should be done before NumPy is imported
for env in ["MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ[env] = "1"
    del env

# NEP-18 not working. For compatibility of newer NumPy version. See gh-cupy/cupy#2130
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

del os, sys

from renormalizer.utils.log import init_log

init_log()

del init_log
