# -*- coding: utf-8 -*-

from functools import update_wrapper

import numpy as np


from renormalizer.utils.log import DEFAULT_NP_ERRCONFIG
from renormalizer.lib.integrate import _ivp
from renormalizer.lib.integrate.abm import solve_ivp_abm


def solve_ivp(*args, **kwargs):
    with np.errstate(**DEFAULT_NP_ERRCONFIG):
        if "method" in kwargs and kwargs["method"] == "ABM":
            kwargs.pop("method")
            return solve_ivp_abm(*args, **kwargs)
        return _ivp.solve_ivp(*args, **kwargs)


update_wrapper(solve_ivp, _ivp.solve_ivp)
