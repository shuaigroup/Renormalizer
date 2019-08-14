# -*- coding: utf-8 -*-

from ephMPS.tests.parameter import mol_list
from ephMPS.tests.parameter_PBI import construct_mol


def test_symmetry():
    assert not mol_list.is_symmetric
    assert construct_mol(10, 10, 0).is_symmetric


def test_idx():
    assert mol_list.e_idx(0) == 0
    assert mol_list.e_idx(1) == 3
    assert mol_list.switch_scheme(4).e_idx(0) == 2
    assert mol_list.ph_idx(0, 0) == 1
    assert mol_list.ph_idx(0, 1) == 2
    assert mol_list.ph_idx(2, 1) == 8
    assert mol_list.switch_scheme(4).ph_idx(2, 1) == 6
    assert mol_list.switch_scheme(4).ph_idx(1, 0) == 3