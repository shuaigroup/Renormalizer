# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import pickle
import os

import pytest
import numpy as np

from renormalizer.model import MolList, Mol, Phonon
from renormalizer.mps import Mpo, Mps, solver
from renormalizer.tests.parameter import mol_list
from renormalizer.mps.tests import cur_dir
from renormalizer.utils import Quantity


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
    assert mpo1.is_hermitian()
    f1 = mpo1.full_operator()
    evals1, _ = np.linalg.eigh(f1)
    offset = Quantity(0.123)
    mpo2 = Mpo(mlist, offset=offset)
    f2 = mpo2.full_operator()
    evals2, _ = np.linalg.eigh(f2)
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
    assert mpo4.is_hermitian()
    # for debugging
    f = mpo4.full_operator()
    mpo3 = Mpo(mlist2)
    assert mpo3.is_hermitian()
    # makeup two states
    mps4 = Mps()
    mps4.mol_list = mlist1
    mps4.use_dummy_qn = True
    mps4.append(np.array([1, 0]).reshape((1,2,1)))
    mps4.append(np.array([0, 0, 1]).reshape((1,-1,1)))
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


@pytest.mark.parametrize("scheme", (2, 3, 4))
def test_intersite(scheme):

    local_mlist = mol_list.switch_scheme(scheme)

    mpo1 = Mpo.intersite(local_mlist, {0:r"a^\dagger"}, {}, Quantity(1.0))
    mpo2 = Mpo.onsite(local_mlist, r"a^\dagger", mol_idx_set=[0])
    assert mpo1.distance(mpo2) == pytest.approx(0, abs=1e-5)

    mpo3 = Mpo.intersite(local_mlist, {2:r"a^\dagger a"}, {}, Quantity(1.0))
    mpo4 = Mpo.onsite(local_mlist, r"a^\dagger a", mol_idx_set=[2])
    assert mpo3.distance(mpo4) == pytest.approx(0, abs=1e-5)

    mpo5 = Mpo.intersite(local_mlist, {2:r"a^\dagger a"}, {}, Quantity(0.5))
    assert mpo5.add(mpo5).distance(mpo4) == pytest.approx(0, abs=1e-5)

    mpo6 = Mpo.intersite(local_mlist, {0:r"a^\dagger",2:"a"}, {}, Quantity(1.0))
    mpo7 = Mpo.onsite(local_mlist, "a", mol_idx_set=[2])
    assert mpo2.apply(mpo7).distance(mpo6) == pytest.approx(0, abs=1e-5)

    # the tests are based on the similarity between scheme 2 and scheme 3
    # so scheme 3 and scheme 4 will be skipped
    if scheme == 2:
        mpo8 = Mpo(local_mlist)
        # a dirty hack to switch from scheme 2 to scheme 3
        test_mlist = local_mlist.switch_scheme(2)
        test_mlist.scheme = 3
        mpo9 = Mpo(test_mlist)
        mpo10 = Mpo.intersite(test_mlist, {0:r"a^\dagger",2:"a"}, {},
                Quantity(local_mlist.j_matrix[0,2]))
        mpo11 = Mpo.intersite(test_mlist, {2:r"a^\dagger",0:"a"}, {},
                Quantity(local_mlist.j_matrix[0,2]))

        assert mpo11.conj_trans().distance(mpo10) == pytest.approx(0, abs=1e-6)
        assert mpo8.distance(mpo9 + mpo10 + mpo11) == pytest.approx(0, abs=1e-6)

        test_mlist.period = True
        mpo12 = Mpo(test_mlist)
        assert mpo12.distance(mpo9 + mpo10 + mpo11) == pytest.approx(0, abs=1e-6)

    ph_mpo1 = Mpo.ph_onsite(local_mlist, "b", 1, 1)
    ph_mpo2 = Mpo.intersite(local_mlist, {}, {(1,1):"b"})
    assert ph_mpo1.distance(ph_mpo2) == pytest.approx(0, abs=1e-6)


def test_phonon_onsite():
    gs = Mps.gs(mol_list, max_entangled=False)
    assert not gs.ph_occupations.any()
    b2 = Mpo.ph_onsite(mol_list, r"b^\dagger", 0, 0)
    p1 = b2.apply(gs).normalize()
    assert np.allclose(p1.ph_occupations, [1, 0, 0, 0, 0, 0])
    p2 = b2.apply(p1).normalize()
    assert np.allclose(p2.ph_occupations, [2, 0, 0, 0, 0, 0])
    b = b2.conj_trans()
    assert b.distance(Mpo.ph_onsite(mol_list, r"b", 0, 0)) == 0
    assert b.apply(p2).normalize().distance(p1) == pytest.approx(0, abs=1e-5)
