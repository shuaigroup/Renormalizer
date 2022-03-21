# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
import numpy as np

from renormalizer.utils.rk import RungeKutta


def test_rk():
    std_list = [
        ("Forward_Euler", [1.0, 1.0]),
        ("Heun_RK2", [1.0, 1.0, 0.5]),
        ("Ralston_RK2", [1.0, 1.0, 0.5]),
        ("midpoint_RK2", [1.0, 1.0, 0.5]),
        ("Kutta_RK3", [1.0, 1.0, 0.5, 0.16666667]),
        ("C_RK4", [1.0, 1.0, 0.5, 0.16666667, 0.04166667]),
        ("38rule_RK4", [1.0, 1.0, 0.5, 0.16666667, 0.04166667]),
    ]
    for method, std in std_list:
        assert np.allclose(RungeKutta(method).runge_kutta_ti_coefficient(), std)
