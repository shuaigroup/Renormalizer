# -*- coding: utf-8 -*-

from ephMPS.tests.parameter import mol_list
from ephMPS.tests.parameter_PBI import construct_mol


def test_symmetry():
    assert not mol_list.is_symmetric
    assert construct_mol(10, 10, 0).is_symmetric