# -*- coding: utf-8 -*-

from renormalizer.model.mlist import ModelTranslator, MolList2
from renormalizer.utils.basis import BasisSimpleElectron, BasisSHO
from renormalizer.tests.parameter import mol_list


def test_idx():
    assert mol_list.e_idx(0) == 0
    assert mol_list.e_idx(1) == 3
    assert mol_list.switch_scheme(4).e_idx(0) == 2
    assert mol_list.ph_idx(0, 0) == 1
    assert mol_list.ph_idx(0, 1) == 2
    assert mol_list.ph_idx(2, 1) == 8
    assert mol_list.switch_scheme(4).ph_idx(2, 1) == 6
    assert mol_list.switch_scheme(4).ph_idx(1, 0) == 3


def test_mollist2():
    # 10 mols
    n = 10
    # 5 modes per mol
    m = 5
    order_list = []
    basis_list = []
    order_dict = {}
    basis_dict = {}
    for ni in range(n):
        dof = f"e_{ni}"
        order_list.append(dof)
        basis = BasisSimpleElectron()
        basis_list.append(basis)

        order_dict[dof] = len(order_list) - 1
        basis_dict[dof] = basis

        for mi in range(m):
            dof = f"v_{ni * n + mi}"
            order_list.append(dof)
            basis = BasisSHO(1, 4)
            basis_list.append(basis)

            order_dict[dof] = len(order_list) - 1
            basis_dict[dof] = basis

    mlist1 = MolList2(order_list, basis_list, {}, ModelTranslator.general_model)
    mlist2 = MolList2(order_list, basis_dict, {}, ModelTranslator.general_model)
    mlist3 = MolList2(order_dict, basis_list, {}, ModelTranslator.general_model)
    mlist4 = MolList2(order_dict, basis_dict, {}, ModelTranslator.general_model)

    def check_eq(m1: MolList2, m2: MolList2):
        assert m1.order == m2.order
        assert m1.basis == m2.basis

    check_eq(mlist1, mlist2)
    check_eq(mlist1, mlist3)
    check_eq(mlist1, mlist4)