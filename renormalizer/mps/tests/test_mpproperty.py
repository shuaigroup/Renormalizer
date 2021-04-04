# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

import pytest

from renormalizer.mps import Mps, Mpo, MpDm, ThermalProp
from renormalizer.mps.backend import np
from renormalizer.tests.parameter import holstein_model
from renormalizer.utils import Quantity


creation_operator = Mpo.onsite(
    holstein_model, r"a^\dagger", dof_set={holstein_model.mol_num // 2}
)


def check_property(mp):
    electron_occupation = np.zeros((holstein_model.mol_num))
    electron_occupation[holstein_model.mol_num // 2] = 1
    assert mp.norm == pytest.approx(1)
    assert np.allclose(mp.e_occupations, electron_occupation)
    assert np.allclose(mp.ph_occupations, 0)


def test_mps():
    gs_mps = Mps.ground_state(holstein_model, max_entangled=False)
    mps = creation_operator @ gs_mps
    check_property(mps)


def test_mpo():
    gs_dm = MpDm.max_entangled_gs(holstein_model)
    beta = Quantity(10, "K").to_beta()
    tp = ThermalProp(gs_dm, exact=True, space="GS")
    tp.evolve(None, 500, beta / 1j)
    gs_dm = tp.latest_mps
    mp = creation_operator @ gs_dm
    check_property(mp)
