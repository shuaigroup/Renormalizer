# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import pytest

from renormalizer.mps.gs import construct_mps_mpo_2, optimize_mps
from renormalizer.tests.parameter import mol_list, custom_mol_list
from renormalizer.utils import constant

nexciton = 1
procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]


@pytest.mark.parametrize("scheme", (
        1,
        4,
))
def test_optimization(scheme):
    mps, mpo = construct_mps_mpo_2(mol_list.switch_scheme(scheme), procedure[0][0], nexciton)
    mps.optimize_config.procedure = procedure
    mps.optimize_config.method = "2site"
    energies, _ = optimize_mps(mps.copy(), mpo)
    assert energies[-1] == pytest.approx(0.08401412 + mol_list.gs_zpe, rel=1e-5)

    mps.optimize_config.method = "1site"
    energies, _ = optimize_mps(mps.copy(), mpo)
    assert energies[-1] == pytest.approx(0.08401412 + mol_list.gs_zpe, rel=1e-5)

@pytest.mark.parametrize("method", (
        "1site",
        "2site",
))
def test_multistate(method):
    mps, mpo = construct_mps_mpo_2(mol_list, procedure[0][0], nexciton)
    mps.optimize_config.procedure = procedure
    mps.optimize_config.nroots = 4
    mps.optimize_config.method = method
    mps.optimize_config.e_atol = 1e-6
    mps.optimize_config.e_rtol = 1e-6
    energies, _ = optimize_mps(mps.copy(), mpo)
    energy_std = np.array([0.08401412, 0.08449771, 0.08449801, 0.08449945]) + mol_list.gs_zpe
    assert np.allclose(energies[-1], energy_std)
