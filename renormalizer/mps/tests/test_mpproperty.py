# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

import numpy as np
import pytest

from renormalizer.mps import Mps, Mpo, MpDm, ThermalProp
from renormalizer.tests.parameter import mol_list
from renormalizer.utils import Quantity


creation_operator = Mpo.onsite(
    mol_list, r"a^\dagger", mol_idx_set={mol_list.mol_num // 2}
)


def check_property(mp):
    electron_occupation = np.zeros((mol_list.mol_num))
    electron_occupation[mol_list.mol_num // 2] = 1
    assert mp.norm == pytest.approx(1)
    assert mp.r_square == pytest.approx(0)
    assert np.allclose(mp.e_occupations, electron_occupation)
    assert np.allclose(mp.ph_occupations, np.zeros((mol_list.ph_modes_num)))


def test_mps():
    gs_mps = Mps.gs(mol_list, max_entangled=False)
    mps = creation_operator @ gs_mps
    check_property(mps)


def test_mpo():
    gs_dm = MpDm.max_entangled_gs(mol_list)
    beta = Quantity(10, "K").to_beta()
    tp = ThermalProp(gs_dm, Mpo(gs_dm.mol_list), exact=True, space="GS")
    tp.evolve(None, 500, beta / 1j)
    gs_dm = tp.latest_mps
    mp = creation_operator @ gs_dm
    check_property(mp)
