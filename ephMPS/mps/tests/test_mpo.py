# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import pickle
import os

import pytest
import numpy as np

from ephMPS.model import MolList, Mol, Phonon
from ephMPS.mps import Mpo, Mps, solver
from ephMPS.tests.parameter import mol_list
from ephMPS.mps.tests import cur_dir
from ephMPS.utils import Quantity


@pytest.mark.parametrize("dt, space, shift", ([30, "GS", 0.0], [30, "EX", 0.0]))
def test_exact_propagator(dt, space, shift):
    prop_mpo = Mpo.exact_propagator(mol_list, -1.0j * dt, space, shift)
    with open(os.path.join(cur_dir, "test_exact_propagator.pickle"), "rb") as fin:
        std_dict = pickle.load(fin)
    std_mpo = std_dict[space]
    assert prop_mpo == std_mpo

@pytest.mark.parametrize("scheme", (1, 2, 3, 4))
def test_offset(scheme):
    ph = Phonon.simple_phonon(Quantity(3.33), Quantity(1), 2)
    m = Mol(Quantity(0), [ph] * 2)
    mlist = MolList([m] * 2, Quantity(17), scheme=scheme)
    mpo1 = Mpo(mlist)
    f1 = mpo1.full_operator()
    evals1, _ = np.linalg.eigh(f1.array)
    offset = Quantity(0.123)
    mpo2 = Mpo(mlist, offset=offset)
    f2 = mpo2.full_operator()
    evals2, _ = np.linalg.eigh(f2.array)
    assert np.allclose(evals1 - offset.as_au(), evals2)

def test_identity():
    identity = Mpo.identity(mol_list)
    mps = Mps.random(mol_list, nexciton=1, m_max=5)
    assert mps.expectation(identity) == pytest.approx(mps.dmrg_norm) == pytest.approx(1)


def test_scheme4():
    ph = Phonon.simple_phonon(Quantity(3.33), Quantity(1), 2)
    m1 = Mol(Quantity(0), [ph])
    m2 = Mol(Quantity(0), [ph]*2)
    mlist1 = MolList([m1, m2], Quantity(17), 4)
    mlist2 = MolList([m1, m2], Quantity(17), 3)
    mpo4 = Mpo(mlist1)
    # for debugging
    f = mpo4.full_operator()
    mpo3 = Mpo(mlist2)
    # makeup two states
    mps4 = Mps()
    mps4.mol_list = mlist1
    mps4.use_dummy_qn = True
    mps4.append(np.array([1, 0]).reshape((1,2,1)))
    mps4.append(np.array([0, 1]).reshape((1,2,1)))
    mps4.append(np.array([0.707, 0.707]).reshape((1,2,1)))
    mps4.append(np.array([1, 0]).reshape((1,2,1)))
    mps4.build_empty_qn()
    e4 = mps4.expectation(mpo4)
    mps3 = Mps()
    mps3.mol_list = mlist2
    mps3.append(np.array([1, 0]).reshape((1,2,1)))
    mps3.append(np.array([1, 0]).reshape((1,2,1)))
    mps3.append(np.array([0, 1]).reshape((1,2,1)))
    mps3.append(np.array([0.707, 0.707]).reshape((1,2,1)))
    mps3.append(np.array([1, 0]).reshape((1,2,1)))
    e3 = mps3.expectation(mpo3)
    assert pytest.approx(e4) == e3