# -*- coding: utf-8 -*-

from math import sqrt

import pytest
import numpy as np

from ephMPS.model.phonon import Phonon
from ephMPS.utils import Quantity


def test_property():
    ph = Phonon.simple_phonon(
        omega=Quantity(1, "a.u."), displacement=Quantity(1, "a.u."), n_phys_dim=10
    )
    assert ph.reorganization_energy.as_au() == pytest.approx(0.5)
    assert ph.coupling_constant == pytest.approx(sqrt(0.5))
    evecs = ph.get_displacement_evecs()
    s = 0.5
    res = [np.exp(-s)]
    for k in range(1, 10):
        res.append(res[-1] * s / k)
    assert np.allclose(res, evecs[:, 0] ** 2)
