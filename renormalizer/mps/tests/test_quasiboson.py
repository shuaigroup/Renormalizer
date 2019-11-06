# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import pytest

from renormalizer.mps.matrix import einsum
from renormalizer.mps import Mpo, solver
from renormalizer.tests import parameter
from renormalizer.utils import constant

nexciton = 1
procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]


def test_quasiboson_constructMPO():

    mol_list1 = parameter.custom_mol_list(None, [4, 4])
    mps1, mpo1 = solver.construct_mps_mpo_2(
        mol_list1, procedure[0][0], nexciton
    )

    mol_list2 = mol_list = parameter.custom_mol_list(None, [4, 4], [2, 2], [1e-7, 1e-7])
    mps2, mpo2 = solver.construct_mps_mpo_2(
        mol_list2, procedure[0][0], nexciton
    )

    # merge the decomposed MPO
    mpo_merge = Mpo()
    mpo_merge.mol_list = mol_list
    impo = 0
    for mol in mol_list:
        mpo_merge.append(mpo2[impo])
        impo += 1
        for _ in mol.dmrg_phs:
            mo = einsum("abcd, defg -> abecfg", mpo2[impo], mpo2[impo + 1]).reshape(
                (mpo2[impo].shape[0], 4, 4, mpo2[impo + 1].shape[-1])
            )
            mpo_merge.append(mo)
            impo += 2

    assert mpo1.distance(mpo_merge) == pytest.approx(0)

    mps2.optimize_config.procedure = procedure
    mps2.optimize_config.method = "2site"
    energy = solver.optimize_mps(mps2, mpo2)
    assert np.min(energy) * constant.au2ev == pytest.approx(2.28614053133, rel=1e-4)

    mps2.optimize_config.method = "1site"
    energy = solver.optimize_mps(mps2, mpo2)
    assert np.min(energy) * constant.au2ev == pytest.approx(2.28614053133, rel=1e-4)


@pytest.mark.parametrize(
    "value",
    (
        [
            [[64, 64]],
            [[64, 64], [6, 6], [1e-7, 1e-7]],
            [[64, 64], [6, 1], [1e-7, 1e-7]],
        ],
        [
            [[27, 27]],
            [[27, 27], [3, 3], [1e-7, 1e-7]],
            [[27, 27], [3, 1], [1e-7, 1e-7]],
        ],
    ),
)
def test_quasiboson_solver(value):
    np.random.seed(0)
    # normal boson
    mol_list1 = parameter.custom_mol_list(None, *value[0])
    mps1, mpo1 = solver.construct_mps_mpo_2(
        mol_list1, procedure[0][0], nexciton
    )
    mps1.optimize_config.procedure = procedure

    # quasiboson
    mol_list2 = parameter.custom_mol_list(None, *value[1])
    mps2, mpo2 = solver.construct_mps_mpo_2(
        mol_list2, procedure[0][0], nexciton
    )
    mps2.optimize_config.procedure = procedure

    # quasiboson + normal boson
    mol_list3 = parameter.custom_mol_list(None, *value[2])
    mps3, mpo3 = solver.construct_mps_mpo_2(
        mol_list3, procedure[0][0], nexciton
    )
    mps3.optimize_config.procedure = procedure

    for method in ["1site", "2site"]:
        mps1.optimize_config.method = method
        mps2.optimize_config.method = method
        mps3.optimize_config.method = method
        energy1 = solver.optimize_mps(mps1, mpo1)
        energy2 = solver.optimize_mps(mps2, mpo2)
        energy3 = solver.optimize_mps(mps3, mpo3)
        assert np.min(energy1) == pytest.approx(np.min(energy2), rel=1e-4)
        assert np.min(energy2) == pytest.approx(np.min(energy3), rel=1e-4)
