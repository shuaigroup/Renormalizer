# -*- coding: utf-8 -*-

import numpy as np
import pytest

from renormalizer.mps import Mps, Mpo
from renormalizer.model import MolList2, ModelTranslator
from renormalizer.utils.basis import BasisSHO, BasisMultiElectronVac, BasisMultiElectron, BasisSimpleElectron, Op
from renormalizer.tests import parameter


@pytest.mark.parametrize("mpos", (
        [
            Mpo.onsite(parameter.mol_list, r"a^\dagger a", mol_idx_set={i})
            for i in range(parameter.mol_list.mol_num)
        ],
        [
            Mpo.intersite(parameter.mol_list, {i: "a", i + 1: r"a^\dagger"}, {})
            for i in range(parameter.mol_list.mol_num - 1)
        ],
        [
            Mpo.intersite(parameter.mol_list, {i: "a", i + 1: r"a^\dagger"}, {})
            for i in range(parameter.mol_list.mol_num - 1)
        ] + [
            Mpo.intersite(parameter.mol_list, {i: "a"}, {})
            for i in range(parameter.mol_list.mol_num - 1)
        ]
))
def test_expectations(mpos):
    random = Mps.random(parameter.mol_list, 1, 20)

    e1 = random.expectations(mpos)
    e2 = random.expectations(mpos, opt=False)

    assert np.allclose(e1, e2)


def check_reduced_density_matrix(order, basis):
    mol_list = MolList2(order, basis, {}, ModelTranslator.general_model)
    mps = Mps.random(mol_list, 1, 20)
    rdm = mps.calc_reduced_density_matrix().real
    assert np.allclose(np.diag(rdm), mps.e_occupations)
    # only test a sample. Should be enough.
    mpo = Mpo.general_mpo(mol_list, model={(f"e_0", f"e_3"): [(Op(r"a^\dagger", 1), Op("a", -1), 1.0)]},
                          model_translator=ModelTranslator.general_model)
    assert rdm[-1][0] == pytest.approx(mps.expectation(mpo))


def test_reduced_density_matrix():
    # case one: simple electron
    order = {"e_0": 0, "v_0": 1, "e_1": 2, "v_1": 3, "e_2": 4, "v_2": 5, "e_3": 6, "v_3": 7}
    basis = [BasisSimpleElectron(), BasisSHO(1, 2)] * 4
    check_reduced_density_matrix(order, basis)

    # case two: multi electron
    order = {"e_0": 0, "v_0": 1, "e_1": 0, "v_1": 2, "e_2": 0, "v_2": 3, "e_3": 0, "v_3": 4}
    basis = [BasisMultiElectron(4, [1,1,1,1])] + [BasisSHO(1, 2)] * 4
    check_reduced_density_matrix(order, basis)

    # case three: MultiElectronVac on multiple sites
    order = {"e_0": 0, "v_0": 1, "e_1": 0, "v_1": 2, "e_2": 3, "v_2": 4, "e_3": 3, "v_3": 5}
    basis = [BasisMultiElectronVac(2, dof_idx=[0, 1])] + [BasisSHO(1, 2)] * 2 \
            + [BasisMultiElectronVac(2, dof_idx=[2, 3])] +[BasisSHO(1, 2)] * 2
    check_reduced_density_matrix(order, basis)





