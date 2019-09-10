# -*- coding: utf-8 -*-

import os

import numpy as np
import pytest

from renormalizer.mps import Mps, Mpo
from renormalizer.tests import parameter


def test_save_load():
    mps = Mpo.onsite(parameter.mol_list, "a^\dagger", mol_idx_set={0}).apply(Mps.gs(parameter.mol_list, False))
    mpo = Mpo(parameter.mol_list)
    mps1 = mps.copy()
    for i in range(2):
        mps1 = mps1.evolve(mpo, 10)
    mps2 = mps.evolve(mpo, 10)
    fname = "test.npz"
    mps2.dump(fname)
    mps2 = Mps.load(parameter.mol_list, fname)
    mps2 = mps2.evolve(mpo, 10)
    assert np.allclose(mps1.e_occupations, mps2.e_occupations)
    os.remove(fname)


def check_distance(a: Mps ,b: Mps):
    d1 = (a - b).dmrg_norm
    d2 = a.distance(b)
    a_array = a.full_wfn().array
    b_array = b.full_wfn().array
    d3 = np.linalg.norm(a_array - b_array)
    assert d1 == pytest.approx(d2) == pytest.approx(d3)


def test_distance():
    mol_list = parameter.custom_mol_list(n_phys_dim=(2, 2))
    a = Mps.random(mol_list, 1, 10)
    b = Mps.random(mol_list, 1, 10)
    check_distance(a, b)
    h = Mpo(mol_list)
    for i in range(100):
        a = a.evolve(h, 10)
        b = b.evolve(h, 10)
        check_distance(a, b)
