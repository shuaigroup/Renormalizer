# -*- coding: utf-8 -*-

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