# -*- coding: utf-8 -*-

from math import sqrt
from functools import reduce

import pytest
import numpy as np

from ephMPS.model.phonon import Phonon
from ephMPS.utils import Quantity


def test_property():
    ph = Phonon.simple_phonon(
        omega=Quantity(1), displacement=Quantity(1), n_phys_dim=10
    )
    assert ph.reorganization_energy.as_au() == pytest.approx(0.5)
    assert ph.coupling_constant == pytest.approx(sqrt(0.5))
    evecs = ph.get_displacement_evecs()
    s = 0.5
    res = [np.exp(-s)]
    for k in range(1, 10):
        res.append(res[-1] * s / k)
    assert np.allclose(res, evecs[:, 0] ** 2)

    ph2 = Phonon.simple_phonon(
        omega=Quantity(1), displacement=Quantity(1), n_phys_dim=10
    )
    assert ph == ph2


def test_simplest_phonon():
    ph =Phonon.simplest_phonon(Quantity(0.1), Quantity(10))
    assert ph.nlevels == 32
    ph = Phonon.simplest_phonon(Quantity(1), Quantity(1))
    assert ph.nlevels == 16
    ph = Phonon.simplest_phonon(Quantity(0.128), Quantity(6.25))
    assert ph.nlevels == 16
    ph = Phonon.simplest_phonon(Quantity(0.032), Quantity(6.25))
    assert ph.nlevels == 16
    ph = Phonon.simplest_phonon(Quantity(1), Quantity(0.01), temperature=Quantity(1))
    assert ph.nlevels == 14
    ph = Phonon.simplest_phonon(Quantity(520, "cm-1"), Quantity(28, "meV"), Quantity(298, "K"), lam=True)
    assert ph.nlevels == 19


def test_split():
    ph = Phonon.simplest_phonon(Quantity(100, "cm-1"), Quantity(1))
    ph1, ph2 = ph.split(width=Quantity(20, "cm-1"))
    assert ph1.e0 == ph2.e0 == ph.e0 / 2
    assert ph1.omega[0] == Quantity(80, "cm-1").as_au()
    ph_list = ph.split(n=100)
    assert reduce(lambda x, y: x+y, map(lambda x: x.e0, ph_list)) == ph.e0