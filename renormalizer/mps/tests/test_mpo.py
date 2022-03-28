# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import os
import random
import pickle

import numpy as np
import pytest

from renormalizer.model import Mol, Phonon, HolsteinModel, Model, Op
from renormalizer.model.basis import BasisHalfSpin
from renormalizer.mps import Mpo, Mps
from renormalizer.mps.tests import cur_dir
from renormalizer.tests.parameter import holstein_model
from renormalizer.utils import Quantity
from renormalizer.utils.qutip_utils import get_spin_hamiltonian


@pytest.mark.parametrize("nsites", [5, 10])
# More sites make MPO representation not efficient
# Not good for testing
@pytest.mark.parametrize("nterms", [100, 1000])
def test_symbolic_mpo(nsites, nterms):

    possible_operators = [
        "sigma_+",
        "sigma_-",
        "sigma_z"
    ]
    ham_terms = []
    for i in range(nterms):
        op_list = [Op(random.choice(possible_operators), j) for j in range(nsites)]
        ham_terms.append(Op.product(op_list) * random.random())
    basis = [BasisHalfSpin(i) for i in range(nsites)]
    model = Model(basis, ham_terms)
    mpo = Mpo(model)
    dense_mpo = mpo.todense()
    qutip_ham = get_spin_hamiltonian(ham_terms)
    assert np.allclose(dense_mpo, qutip_ham.data.todense())


@pytest.mark.parametrize("nsites", [5, 10])
# More sites make MPO representation not efficient
# Not good for testing
@pytest.mark.parametrize("nterms", [100, 1000])
def test_swap_symbolic_mpo(nsites, nterms):
    possible_operators = [
        "sigma_+",
        "sigma_-",
        "sigma_z"
    ]
    ham_terms = []
    for i in range(nterms):
        op_list = [Op(random.choice(possible_operators), j) for j in range(nsites)]
        ham_terms.append(Op.product(op_list) * random.random())
    basis = [BasisHalfSpin(i) for i in range(nsites)]
    model = Model(basis, ham_terms)
    mpo = Mpo(model)
    for i in range(20):
        isite1 = max(int(random.random() * nsites) - 1, 0)
        isite2 = isite1 + 1
        basis = basis.copy()
        basis[isite1], basis[isite2] = basis[isite2], basis[isite1]
        new_model = Model(basis, ham_terms)
        mpo.try_swap_site(new_model, False)
        ref_mpo = Mpo(new_model)
        mpo_dense = mpo.todense()
        ref_dense = ref_mpo.todense()
        assert np.allclose(mpo_dense, ref_dense)


@pytest.mark.parametrize("dt, space, shift", ([30, "GS", 0.0], [30, "EX", 0.0]))
def test_exact_propagator(dt, space, shift):
    prop_mpo = Mpo.exact_propagator(holstein_model, -1.0j * dt, space, shift)
    with open(os.path.join(cur_dir, "test_exact_propagator.pickle"), "rb") as fin:
        std_dict = pickle.load(fin)
    std_mpo = std_dict[space]
    assert prop_mpo == std_mpo


@pytest.mark.parametrize("scheme", (1, 4))
def test_offset(scheme):
    ph = Phonon.simple_phonon(Quantity(3.33), Quantity(1), 2)
    m = Mol(Quantity(0), [ph] * 2)
    mlist = HolsteinModel([m] * 2, Quantity(17), )
    mpo1 = Mpo(mlist)
    assert mpo1.is_hermitian()
    f1 = mpo1.todense()
    evals1, _ = np.linalg.eigh(f1)
    offset = Quantity(0.123)
    mpo2 = Mpo(mlist, offset=offset)
    f2 = mpo2.todense()
    evals2, _ = np.linalg.eigh(f2)
    assert np.allclose(evals1 - offset.as_au(), evals2)


def test_identity():
    identity = Mpo.identity(holstein_model)
    mps = Mps.random(holstein_model, nexciton=1, m_max=5)
    assert mps.expectation(identity) == pytest.approx(mps.mp_norm) == pytest.approx(1)


def test_scheme4():
    ph = Phonon.simple_phonon(Quantity(3.33), Quantity(1), 2)
    m1 = Mol(Quantity(0), [ph])
    m2 = Mol(Quantity(0), [ph]*2)
    model4 = HolsteinModel([m1, m2], Quantity(17), 4)
    model3 = HolsteinModel([m1, m2], Quantity(17), 3)
    mpo4 = Mpo(model4)
    assert mpo4.is_hermitian()
    # for debugging
    f = mpo4.todense()
    mpo3 = Mpo(model3)
    assert mpo3.is_hermitian()
    # makeup two states
    mps4 = Mps()
    mps4.model = model4
    mps4.append(np.array([1, 0]).reshape((1,2,1)))
    mps4.append(np.array([0, 0, 1]).reshape((1,-1,1)))
    mps4.append(np.array([0.707, 0.707]).reshape((1,2,1)))
    mps4.append(np.array([1, 0]).reshape((1,2,1)))
    mps4.build_empty_qn()
    e4 = mps4.expectation(mpo4)
    mps3 = Mps()
    mps3.model = model3
    mps3.append(np.array([1, 0]).reshape((1,2,1)))
    mps3.append(np.array([1, 0]).reshape((1,2,1)))
    mps3.append(np.array([0, 1]).reshape((1,2,1)))
    mps3.append(np.array([0.707, 0.707]).reshape((1,2,1)))
    mps3.append(np.array([1, 0]).reshape((1,2,1)))
    e3 = mps3.expectation(mpo3)
    assert pytest.approx(e4) == e3


@pytest.mark.parametrize("scheme", (1, 4))
def test_intersite(scheme):

    local_mlist = holstein_model.switch_scheme(scheme)

    mpo1 = Mpo.intersite(local_mlist, {0:r"a^\dagger"}, {}, Quantity(1.0))
    mpo2 = Mpo.onsite(local_mlist, r"a^\dagger", dof_set=[0])
    assert mpo1.distance(mpo2) == pytest.approx(0, abs=1e-5)

    mpo3 = Mpo.intersite(local_mlist, {2:r"a^\dagger a"}, {}, Quantity(1.0))
    mpo4 = Mpo.onsite(local_mlist, r"a^\dagger a", dof_set=[2])
    assert mpo3.distance(mpo4) == pytest.approx(0, abs=1e-5)

    mpo5 = Mpo.intersite(local_mlist, {2:r"a^\dagger a"}, {}, Quantity(0.5))
    assert mpo5.add(mpo5).distance(mpo4) == pytest.approx(0, abs=1e-5)

    mpo6 = Mpo.intersite(local_mlist, {0:r"a^\dagger",2:"a"}, {}, Quantity(1.0))
    mpo7 = Mpo.onsite(local_mlist, "a", dof_set=[2])
    assert mpo2.apply(mpo7).distance(mpo6) == pytest.approx(0, abs=1e-5)

    mpo8 = Mpo.intersite(local_mlist, {0: r"a^\dagger", 2: "a"}, {},
                         Quantity(local_mlist.j_matrix[0,2]))
    mpo9 = Mpo.intersite(local_mlist, {2:r"a^\dagger",0:"a"}, {},
            Quantity(local_mlist.j_matrix[0,2]))

    assert mpo9.conj_trans().distance(mpo8) == pytest.approx(0, abs=1e-6)

    ph_mpo1 = Mpo.ph_onsite(local_mlist, "b", 1, 1)
    ph_mpo2 = Mpo.intersite(local_mlist, {}, {(1,1):"b"})
    assert ph_mpo1.distance(ph_mpo2) == pytest.approx(0, abs=1e-6)


def test_phonon_onsite():
    gs = Mps.ground_state(holstein_model, max_entangled=False)
    assert not gs.ph_occupations.any()
    b2 = Mpo.ph_onsite(holstein_model, r"b^\dagger", 0, 0)
    p1 = b2.apply(gs).normalize("mps_only")
    assert np.allclose(p1.ph_occupations, [1, 0, 0, 0, 0, 0])
    p2 = b2.apply(p1).normalize("mps_only")
    assert np.allclose(p2.ph_occupations, [2, 0, 0, 0, 0, 0])
    b = b2.conj_trans()
    assert b.distance(Mpo.ph_onsite(holstein_model, r"b", 0, 0)) == 0
    assert b.apply(p2).normalize("mps_only").distance(p1) == pytest.approx(0, abs=1e-5)