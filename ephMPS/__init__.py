# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
import os

import numpy as np

from ephMPS.utils.log import init_log

np.seterr(divide='raise', over='raise', under='warn', invalid='raise')

init_log()

for env in ['MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[env] = '1'