# -*- coding: utf-8 -*-

import os

import numpy as np
import pytest

from renormalizer.mps import Mps, Mpo
from renormalizer.mps.matrix import tensordot
from renormalizer.mps.lib import Environ
from renormalizer.tests.parameter import custom_mol_list, mol_list


def test_save_load():
    mol_list = custom_mol_list(hartrees=[True, False])
    mps = Mpo.onsite(mol_list, "a^\dagger", mol_idx_set={0}) @ Mps.gs(mol_list, False)
    mpo = Mpo(mol_list)
    mps1 = mps.copy()
    for i in range(2):
        mps1 = mps1.evolve(mpo, 10)
    mps2 = mps.evolve(mpo, 10)
    fname = "test.npz"
    mps2.dump(fname)
    mps2 = Mps.load(mol_list, fname)
    mps2 = mps2.evolve(mpo, 10)
    assert np.allclose(mps1.e_occupations, mps2.e_occupations)
    os.remove(fname)


def check_distance(a: Mps, b: Mps):
    d1 = (a - b).dmrg_norm
    d2 = a.distance(b)
    a_array = a.full_wfn()
    b_array = b.full_wfn()
    d3 = np.linalg.norm(a_array - b_array)
    assert d1 == pytest.approx(d2) == pytest.approx(d3)


def test_distance():
    mol_list = custom_mol_list(n_phys_dim=(2, 2))
    a = Mps.random(mol_list, 1, 10)
    b = Mps.random(mol_list, 1, 10)
    check_distance(a, b)
    h = Mpo(mol_list)
    for i in range(100):
        a = a.evolve(h, 10)
        b = b.evolve(h, 10)
        check_distance(a, b)


def test_environ():
    mps = Mps.random(mol_list, 1, 10)
    mpo = Mpo(mol_list)
    mps = mps.evolve(mpo, 10)
    environ = Environ(mps, mpo)
    for i in range(len(mps)-1):
        l = environ.read("L", i)
        r = environ.read("R", i+1)
        e = complex(tensordot(l, r, axes=((0, 1, 2), (0, 1, 2)))).real
        assert pytest.approx(e) == mps.expectation(mpo)

