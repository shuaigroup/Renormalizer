# -*- coding: utf-8 -*-

from collections import deque
import logging

from renormalizer.lib.integrate._ivp import solve_ivp
from renormalizer.lib.integrate._ivp.ivp import OdeResult
from renormalizer.lib.integrate._ivp.common import (
    select_initial_step,
    validate_first_step,
    norm,
)

import numpy as np

from renormalizer.mps.backend import xp, backend
from renormalizer.utils.configs import EvolveConfig


logger = logging.getLogger(__file__)

# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.


ORDER = 6
B_COEF = [4277/1440, -2641/480, 4991/720, -3649/720,+959/480, -95/288]
C1 = 19087 / 60480
M_COEF = [95/288, 1427/1440, -133/240, 241/720,-173/1440, 3/160]
C2 = -863 / 60480


def abm_constant_step_size(fun, t0, y0, h, n_steps, atol, rtol, derivatives=None):
    if derivatives is None:
        backward_timespan = [t0 - i * h for i in range(ORDER)]
        backward_sol = solve_ivp(fun, [backward_timespan[0], backward_timespan[-1]], y0, rtol=rtol*1e-2, atol=atol*1e-2, t_eval=backward_timespan)
        derivatives = deque([fun(backward_timespan[-i], backward_sol.y[:, -i]) for i in range(1, ORDER+1)])
    assert len(derivatives) == ORDER

    t, y = t0, y0
    for i in range(n_steps):
        t += h
        py = y.copy()
        for j in range(ORDER):
            py += (h * B_COEF[j]) * derivatives[-j - 1]
        y = y + (h * M_COEF[0]) * fun(t, py)
        for j in range(1, ORDER):
            y += (h * M_COEF[j]) * derivatives[-j]

        derivatives.popleft()
        derivatives.append(fun(t, y))

    #scale = atol + xp.maximum(xp.abs(y), xp.abs(py)) * rtol
    rtol_array = xp.maximum(rtol, atol / xp.maximum(xp.abs(y), xp.abs(py)))
    q_array = (abs((C1 - C2) / C2 * rtol_array * h / xp.abs(py - y)) ** (1 / ORDER)) * SAFETY
    q = min(q_array)
    q = min(q, MAX_FACTOR)
    q = max(q, MIN_FACTOR)

    return y, h*q, derivatives


def solve_ivp_abm(_fun,
                  t_span,
                  y0,
                  evolve_config: EvolveConfig):
    # transform arguments
    t0, tf = float(t_span[0]), float(t_span[1])
    y0 = xp.asarray(y0)
    nfev = 0
    def fun(t, y):
        nonlocal nfev
        nfev += 1
        return _fun(t, y)

    assert tf != t0

    first_step = evolve_config.tdvp_vmf_suggest_h
    adaptive = evolve_config.abm_adaptive
    atol = evolve_config.ivp_atol
    rtol = evolve_config.ivp_rtol


    # determine the first h
    direction = np.sign(tf - t0)
    if first_step is None:
        assert adaptive
        h_abs_non_dividable = select_initial_step(fun, t0, y0, fun(t0, y0), direction, ORDER, rtol, rtol * 1e-3)
    else:
        h_abs_non_dividable = validate_first_step(first_step, t0, tf)
    h_non_dividable = direction * h_abs_non_dividable

    # adjust step-size every 48 steps
    t, y = t0, y0
    total_steps = 0
    if adaptive:
        while not np.isclose(t, tf):
            steps = int(np.round(abs((tf - t) / h_non_dividable)))
            if t == t0 and steps < 36:
                logger.warning("Too few steps in ABM integrator")
                steps = 36
                h = (tf - t) / steps
            elif steps < 60:
                # end of the loop
                h = (tf - t) / steps
            else:
                steps = 48
                h = h_non_dividable
            y, h_non_dividable, _ = abm_constant_step_size(fun, t, y, h, steps, atol, rtol)
            t += steps * h
            total_steps += steps
        evolve_config.tdvp_vmf_suggest_h = abs(h_non_dividable)
    else:
        total_steps = int((tf - t) / h_non_dividable)
        h = h_non_dividable
        assert np.isclose(h * total_steps, tf - t)
        y, h_non_dividable, derivatives = abm_constant_step_size(fun, t, y, h, total_steps, atol, rtol, evolve_config.abm_derivatives)
        evolve_config.abm_derivatives = derivatives

    ys = xp.vstack([y]).T

    return OdeResult(
        y=ys,
        nfev=nfev,
        nsteps=total_steps,
    )