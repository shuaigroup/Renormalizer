# -*- coding: utf-8 -*-

from functools import update_wrapper

import numpy as np
import scipy.integrate

from ephMPS.utils.log import DEFAULT_NP_ERRCONFIG


def solve_ivp(*args, **kwargs):
    with np.errstate(**DEFAULT_NP_ERRCONFIG):
        return scipy.integrate.solve_ivp(*args, **kwargs)


update_wrapper(solve_ivp, scipy.integrate.solve_ivp)
