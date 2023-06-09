# -*- coding: utf-8 -*-

from renormalizer.tests.parameter import holstein_model


def test_idx():
    assert holstein_model.order[0] == 0
    assert holstein_model.order[1] == 3
    assert holstein_model.switch_scheme(4).order[0] == 2
    assert holstein_model.order[(0, 0)] == 1
    assert holstein_model.order[(0, 1)] == 2
    assert holstein_model.order[(2, 1)] == 8
    assert holstein_model.switch_scheme(4).order[(2, 1)] == 6
    assert holstein_model.switch_scheme(4).order[(1, 0)] == 3
