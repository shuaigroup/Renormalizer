# -*- coding: utf-8 -*-

import numpy as np
import pytest

from renormalizer.lib.integrate import solve_ivp
from renormalizer.utils.configs import EvolveConfig


@pytest.mark.parametrize("x0, xf, y0", (
        (0., 10., 2.),
        (10., 0., 123.)
))
@pytest.mark.parametrize("imag", (False, True))
def test_abm(x0, xf, y0, imag):

    def f(x, y):
        return y - x

    if imag:
        y0 = y0 + 1j * y0

    sol1 = solve_ivp(f, [x0, xf], [y0], method="ABM", evolve_config=EvolveConfig(adaptive=True))
    sol2 = solve_ivp(f, [x0, xf], [y0], atol=1e-5, rtol=1e-8)

    assert np.allclose(sol1.y[:, -1], sol2.y[:, -1])