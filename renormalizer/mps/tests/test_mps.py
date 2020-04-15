# -*- coding: utf-8 -*-

import numpy as np
import pytest

from renormalizer.mps import Mps, Mpo
from renormalizer.tests import parameter


@pytest.mark.parametrize("mpos", (
        [
            Mpo.onsite(parameter.mol_list, r"a^\dagger a", mol_idx_set={i})
            for i in range(parameter.mol_list.mol_num)
        ],
        [
            Mpo.intersite(parameter.mol_list, {i: "a", i + 1: r"a^\dagger"}, {})
            for i in range(parameter.mol_list.mol_num - 1)
        ]
))
def test_expectations(mpos):
    random = Mps.random(parameter.mol_list, 1, 20)

    e1 = random.expectations(mpos)
    e2 = random.expectations(mpos, opt=False)

    assert np.allclose(e1, e2)
