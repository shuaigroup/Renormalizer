# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import pytest

from ephMPS.mps import solver
from ephMPS.tests import parameter


@pytest.mark.parametrize(
    "mol_list, target",
    (
        [parameter.hybrid_mol_list, 0.084015672468],
        [parameter.mol_list, 0.08401411562239858],
    ),
)
def test_hybrid_DMRG_H_SCF(mol_list, target):

    nexciton = 1
    mps, mpo = solver.construct_mps_mpo_2(mol_list, 10, nexciton)
    Etot = solver.optimize_mps(mps, mpo)
    # print("Etot", Etot)
    assert Etot == pytest.approx(target, abs=1e-5)

    nexciton = 0
    mps, mpo = solver.construct_mps_mpo_2(mol_list, 10, nexciton)
    Etot = solver.optimize_mps(mps, mpo)
    # print("Etot", Etot)
    assert Etot == pytest.approx(0.0, abs=1e-5)
