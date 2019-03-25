# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import pickle
import os

import pytest

from ephMPS.mps import Mpo
from ephMPS.tests.parameter import mol_list
from ephMPS.mps.tests import cur_dir


@pytest.mark.parametrize("dt, space, shift", ([30, "GS", 0.0], [30, "EX", 0.0]))
def test_exact_propagator(dt, space, shift):
    prop_mpo = Mpo.exact_propagator(mol_list, -1.0j * dt, space, shift)
    with open(os.path.join(cur_dir, "test_exact_propagator.pickle"), "rb") as fin:
        std_dict = pickle.load(fin)
    std_mpo = std_dict[space]
    assert prop_mpo == std_mpo
