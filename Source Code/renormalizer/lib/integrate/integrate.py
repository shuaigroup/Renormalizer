# -*- coding: utf-8 -*-

from functools import update_wrapper

import numpy as np


from renormalizer.utils.log import DEFAULT_NP_ERRCONFIG
from renormalizer.lib.integrate import _ivp


def solve_ivp(*args, **kwargs):
    with np.errstate(**DEFAULT_NP_ERRCONFIG):
        return _ivp.solve_ivp(*args, **kwargs)


update_wrapper(solve_ivp, _ivp.solve_ivp)
