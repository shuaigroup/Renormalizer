# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import pytest

from renormalizer.mps.gs import construct_mps_mpo_2, optimize_mps
from renormalizer.tests.parameter import mol_list, custom_mol_list
from renormalizer.utils import constant

nexciton = 1
procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]


def test_construct_MPO():
    Mmax = 10
    mps1, mpo1 = construct_mps_mpo_2(mol_list.switch_scheme(1), Mmax, nexciton)
    mps2, mpo2 = construct_mps_mpo_2(mol_list, Mmax, nexciton)

    assert mpo1.ephtable == mpo2.ephtable
    assert mpo1.pbond_list == mpo2.pbond_list
    # for double precision the abs could be near 0. In single precision
    # the norm of mpo2 is not correct. (mpo2.dot(mpo2) != mpo1.dot(mpo1)
    # but mpo1.dot(mpo1) == mpo1.dot(mpo2)). Reason unknown
    assert mpo1.distance(mpo2) == pytest.approx(0, abs=1e-3)


def test_construct_MPO_scheme3():
    Mmax = 10
    J = (
        np.array([[0.0, -0.1, 0.0], [-0.1, 0.0, -0.3], [0.0, -0.3, 0.0]])
        / constant.au2ev
    )
    mol_list = custom_mol_list(J)
    mps2, mpo2 = construct_mps_mpo_2(mol_list, Mmax, nexciton)
    mps3, mpo3 = construct_mps_mpo_2(mol_list.switch_scheme(3), Mmax, nexciton)
    assert mpo2.ephtable == mpo3.ephtable
    assert mpo2.pbond_list == mpo3.pbond_list
    assert mpo3.distance(mpo2) == pytest.approx(0)


@pytest.mark.parametrize("scheme", (
        1,
        2,
        4,
))
def test_optimization(scheme):
    mps, mpo = construct_mps_mpo_2(mol_list.switch_scheme(scheme), procedure[0][0], nexciton)
    mps.optimize_config.procedure = procedure
    mps.optimize_config.method = "2site"
    energy = optimize_mps(mps.copy(), mpo)
    assert energy * constant.au2ev == pytest.approx(2.28614053133, rel=1e-5)

    mps.optimize_config.method = "1site"
    energy = optimize_mps(mps.copy(), mpo)
    assert energy * constant.au2ev == pytest.approx(2.28614053133, rel=1e-5)

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
    energy = optimize_mps(mps.copy(), mpo)
    energy_std = [0.08401412, 0.08449771, 0.08449801, 0.08449945]
    assert np.allclose(energy, energy_std)
