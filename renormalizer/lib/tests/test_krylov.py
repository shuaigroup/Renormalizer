# -*- coding: utf-8 -*-

import pytest
import numpy as np
from scipy.linalg import expm

from renormalizer.mps.backend import xp
from renormalizer.lib import expm_krylov


@pytest.mark.parametrize("N", (
    200,
    800
))
@pytest.mark.parametrize("imag", (True, False))
def test_expm(N, imag):
    a1 = np.random.rand(N, N) / N
    if imag:
        a1 = a1 + np.random.rand(N, N) / N / 1j
    a2 = xp.array(a1)
    v = np.random.rand(N)
    if imag:
        v = v + v / 1j
    res1 = expm(a1) @ v
    res2, _ = expm_krylov(lambda x: a2.dot(x), 1, xp.array(v))
    assert xp.allclose(res1, res2)