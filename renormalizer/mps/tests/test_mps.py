# -*- coding: utf-8 -*-

import numpy as np
import pytest

from renormalizer.model import Model
from renormalizer.model.basis import BasisSHO, BasisMultiElectronVac, BasisMultiElectron, BasisSimpleElectron
from renormalizer.model.op import Op
from renormalizer.mps import Mps, Mpo
from renormalizer.tests import parameter


@pytest.mark.parametrize("mpos", (
        [
            Mpo.onsite(parameter.holstein_model, r"a^\dagger a", dof_set={i})
            for i in range(parameter.holstein_model.mol_num)
        ],
        [
            Mpo.intersite(parameter.holstein_model, {i: "a", i + 1: r"a^\dagger"}, {})
            for i in range(parameter.holstein_model.mol_num - 1)
        ],
        [
            Mpo.intersite(parameter.holstein_model, {i: "a", i + 1: r"a^\dagger"}, {})
            for i in range(parameter.holstein_model.mol_num - 1)
        ] + [
            Mpo.intersite(parameter.holstein_model, {i: "a"}, {})
            for i in range(parameter.holstein_model.mol_num - 1)
        ]
))
def test_expectations(mpos):
    random = Mps.random(parameter.holstein_model, 1, 20)

    e1 = random.expectations(mpos)
    e2 = random.expectations(mpos, opt=False)

    assert np.allclose(e1, e2)

    random2 = Mps.random(parameter.holstein_model, 1, 20)

    e1 = random.expectations(mpos, random2)
    e2 = random.expectations(mpos, random2, opt=False)

    assert np.allclose(e1, e2)


def check_reduced_density_matrix(basis):
    model = Model(basis, [])
    mps = Mps.random(model, 1, 20)
    rdm = mps.calc_edof_rdm().real
    assert np.allclose(np.diag(rdm), mps.e_occupations)
    # only test a sample. Should be enough.
    mpo = Mpo(model, Op(r"a^\dagger a", [0, 3]))
    assert rdm[-1][0] == pytest.approx(mps.expectation(mpo))


def test_reduced_density_matrix():
    # case one: simple electron
    basis = []
    for i in range(4):
        basis.append(BasisSimpleElectron(i))
        basis.append(BasisSHO(f"v_{i}", 1, 2))
    check_reduced_density_matrix(basis)

    # case two: multi electron
    basis = [BasisMultiElectron(list(range(4)), [1,1,1,1])] + [BasisSHO(f"v_{i}", 1, 2) for i in range(4)]
    check_reduced_density_matrix(basis)

    # case three: MultiElectronVac on multiple sites
    basis = [BasisMultiElectronVac([0, 1]), BasisSHO("v0", 1, 2), BasisSHO("v1", 1, 2),
             BasisMultiElectronVac([2, 3]), BasisSHO("v2", 1, 2), BasisSHO("v3", 1, 2)]
    check_reduced_density_matrix(basis)


def test_site_entropy():
    mps = Mps.random(parameter.holstein_model, 1, 20)
    mps.canonicalise().normalize("mps_only")
    entropy_1site = mps.calc_entropy("1site")
    entropy_2site = mps.calc_entropy("2site")
    entropy_bond = mps.calc_entropy("bond")
    entropy_mutual = mps.calc_entropy("mutual")
    assert np.allclose(entropy_bond[0], entropy_1site[0])
    assert np.allclose(entropy_bond[-1], entropy_1site[mps.site_num-1])
    assert np.allclose(entropy_bond[1], entropy_2site[(0,1)])
    assert np.allclose(entropy_bond[-2], entropy_2site[(mps.site_num-2, mps.site_num-1)])
    assert np.allclose(entropy_mutual[0,1],
            (entropy_1site[0]+entropy_1site[1]-entropy_2site[(0,1)])/2)


def test_load_from_dense_wfn():
    model = Model(basis=[BasisSimpleElectron(i) for i in range(5)], ham_terms=[])
    ref_mps = Mps.random(model, 1, 20)
    dense_wfn = ref_mps.todense()
    loaded_mps = Mps.from_dense(model, dense_wfn)
    assert np.allclose(dense_wfn, loaded_mps.todense())
