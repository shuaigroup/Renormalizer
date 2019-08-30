# -*- coding: utf-8 -*-
from renormalizer.model.phonon import Phonon
from renormalizer.model.mol import Mol
from renormalizer.utils import Quantity


def test_eq():
    ph = Phonon.simple_phonon(
        omega=Quantity(1, "a.u."), displacement=Quantity(1, "a.u."), n_phys_dim=10
    )
    mol1 = Mol(Quantity(0), [ph, ph])
    mol2 = Mol(Quantity(0), [ph, ph])
    mol3 = Mol(Quantity(1), [ph, ph])
    assert mol1 == mol2
    assert mol1 != mol3