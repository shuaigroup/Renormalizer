# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
import os

# set environment variables to limit NumPy cpu usage
# Note that this should be done before NumPy is imported
for env in ["MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ[env] = "1"

del env, os

from ephMPS.utils.log import init_log

init_log()

del init_log