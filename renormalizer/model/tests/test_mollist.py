# -*- coding: utf-8 -*-

from renormalizer.model.model import Model
from renormalizer.utils.basis import BasisSimpleElectron, BasisSHO
from renormalizer.tests.parameter import holstein_model


def test_idx():
    assert holstein_model.order["e_0"] == 0
    assert holstein_model.order["e_1"] == 3
    assert holstein_model.switch_scheme(4).order["e_0"] == 2
    assert holstein_model.order["v_0"] == 1
    assert holstein_model.order["v_1"] == 2
    assert holstein_model.order["v_5"] == 8
    assert holstein_model.switch_scheme(4).order["v_5"] == 6
    assert holstein_model.switch_scheme(4).order["v_2"] == 3


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

    mlist1 = Model(order_list, basis_list, {})
    mlist2 = Model(order_list, basis_dict, {})
    mlist3 = Model(order_dict, basis_list, {})
    mlist4 = Model(order_dict, basis_dict, {})

    def check_eq(m1: Model, m2: Model):
        assert m1.order == m2.order
        assert m1.basis == m2.basis

    check_eq(mlist1, mlist2)
    check_eq(mlist1, mlist3)
    check_eq(mlist1, mlist4)