# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import pytest
import numpy as np
from scipy.linalg import expm, logm

from renormalizer.model import MolList, Mol, Phonon
from renormalizer.mps import Mpo, Mps
from renormalizer.tests.parameter import mol_list
from renormalizer.utils import Quantity


@pytest.mark.parametrize("scheme", (1,2,3,4))
def test_exact_propagator_with_e(scheme):
    ph = Phonon.simple_phonon(Quantity(2), Quantity(1e-10), 2)
    m1 = Mol(Quantity(1), [ph] * 2)
    m2 = Mol(Quantity(2), [ph])
    mlist = MolList([m2, m1, m2], Quantity(1e-10), scheme=scheme)
    for t in np.linspace(1, 5, 10):
        f1 = expm(-1j * t * Mpo(mlist).full_operator())
        f2 = Mpo.exact_propagator(mlist, -1j * t, include_e=True).full_operator()
        assert np.allclose(f1, f2)


@pytest.mark.parametrize("scheme", (1, 2, 3, 4))
def test_offset(scheme):
    ph = Phonon.simple_phonon(Quantity(3.33), Quantity(1), 2)
    m = Mol(Quantity(0), [ph] * 2)
    mlist = MolList([m] * 2, Quantity(17), scheme=scheme)
    mpo1 = Mpo(mlist)
    assert mpo1.is_hermitian()
    f1 = mpo1.full_operator()
    evals1, _ = np.linalg.eigh(f1.asnumpy())
    offset = Quantity(0.123)
    mpo2 = Mpo(mlist, offset=offset)
    f2 = mpo2.full_operator()
    evals2, _ = np.linalg.eigh(f2.asnumpy())
    assert np.allclose(evals1 - offset.as_au(), evals2)


@pytest.mark.parametrize("scheme", (1, 2, 3, 4))
def test_identity(scheme):
    m = MolList(mol_list.mol_list, Quantity(0.01), scheme=scheme)
    identity = Mpo.identity(m)
    mps = Mps.random(m, nexciton=1, m_max=5)
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

def test_e_intersite():
    mpo1 = Mpo.e_intersite(mol_list, {0:r"a^\dagger"}, Quantity(1.0))
    mpo2 = Mpo.onsite(mol_list, r"a^\dagger", mol_idx_set=[0])
    assert mpo1.distance(mpo2) == pytest.approx(0, abs=1e-5)
    
    mpo3 = Mpo.e_intersite(mol_list, {2:r"a^\dagger a"}, Quantity(1.0))
    mpo4 = Mpo.onsite(mol_list, r"a^\dagger a", mol_idx_set=[2])
    assert mpo3.distance(mpo4) == pytest.approx(0, abs=1e-5)
    
    mpo5 = Mpo.e_intersite(mol_list, {2:r"a^\dagger a"}, Quantity(0.5))
    assert mpo5.add(mpo5).distance(mpo4) == pytest.approx(0, abs=1e-5)
    
    mpo6 = Mpo.e_intersite(mol_list, {0:r"a^\dagger",2:"a"}, Quantity(1.0))
    mpo7 = Mpo.onsite(mol_list, "a", mol_idx_set=[2])
    assert mpo2.apply(mpo7).distance(mpo6) == pytest.approx(0, abs=1e-5)
    
    mpo8 = Mpo(mol_list)
    # a dirty hack to switch from scheme 2 to scheme 3
    mol_list1 = mol_list.switch_scheme(2)
    mol_list1.scheme=3
    mpo9 = Mpo(mol_list1)
    mpo10 = Mpo.e_intersite(mol_list1, {0:r"a^\dagger",2:"a"},
            Quantity(mol_list1.j_matrix[0,2]))
    mpo11 = Mpo.e_intersite(mol_list1, {2:r"a^\dagger",0:"a"},
            Quantity(mol_list1.j_matrix[0,2]))
    
    assert mpo8.distance(mpo9.add(mpo10).add(mpo11)) == pytest.approx(0, abs=1e-6)
    assert mpo8.distance(mpo9.add(mpo10).add(mpo10.conj_trans())) == pytest.approx(0, abs=1e-6)


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


@pytest.mark.parametrize("scheme", (1, 2, 3, 4))
def test_interaction_mpo(scheme):
    # as many random values as possible
    ph1 = Phonon.simple_phonon(Quantity(1), Quantity(2), 2)
    ph2 = Phonon.simple_phonon(Quantity(2), Quantity(3), 2)
    ph3 = Phonon.simple_phonon(Quantity(.1), Quantity(.5), 2)
    m1 = Mol(Quantity(1), [ph1, ph3])
    m2 = Mol(Quantity(2), [ph2])
    j = np.array([[0, .5, 0], [.5, 0, .4], [0, .4, 0]])
    mlist = MolList([m2, m1, m2], j, scheme=scheme)
    mlist_inter = MolList([m2, m1, m2], j, scheme=scheme, inter_t=0)
    exact_prop = Mpo.exact_propagator(mlist, 1, include_e=True)
    h0 = logm(exact_prop.full_operator())
    h1 = Mpo(mlist_inter).full_operator()
    h = Mpo(mlist).full_operator()
    assert np.allclose(h1+h0, h)