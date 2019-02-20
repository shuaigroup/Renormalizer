# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import os

import numpy as np
import pytest

from ephMPS.mps import Mps, Mpo, MpDm
from ephMPS.utils import Quantity
from ephMPS.tests import parameter_PBI
from ephMPS.transport.tests import cur_dir


@pytest.mark.parametrize("n_dmrg_phs", (10, 5))
def test_zt(n_dmrg_phs):

    mol_list = parameter_PBI.construct_mol(4, n_dmrg_phs, 10 - n_dmrg_phs)
    mps = Mps.gs(mol_list, False)
    mpo = Mpo(mol_list, scheme=3)
    # create electron
    mps = Mpo.onsite(mol_list, "a^\dagger", mol_idx_set={0}).apply(mps).normalize(1.0)

    # do the evolution
    nsteps = 30
    dt = 30.0

    occ = [mps.e_occupations]
    for i in range(nsteps):
        mps = mps.evolve(mpo, dt)
        occ.append(mps.e_occupations)
    # make it compatible with std data
    occ = np.array(occ[:nsteps]).transpose()

    with open(os.path.join(cur_dir, "ZT_occ" + str(n_dmrg_phs) + ".npy"), "rb") as f:
        std = np.load(f)
    assert np.allclose(occ, std, rtol=1e-2)


@pytest.mark.parametrize("n_dmrg_phs", (10, 5))
def test_FT_dynamics_hybrid_TDDMRG_TDH(n_dmrg_phs):

    mol_list = parameter_PBI.construct_mol(4, n_dmrg_phs, 10 - n_dmrg_phs)
    mpdm = MpDm.max_entangled_gs(mol_list)
    mpo = Mpo(mol_list, scheme=3)
    temperature = Quantity(2000, "K")
    # why divide by 2? in hybrid_TDDMRG_TDH.py FT_DM_hybrid_TDDMRG_TDH, beta is divided by 2
    mpdm = mpdm.thermal_prop_exact(
        mpo, temperature.to_beta() / 2, 1, "GS", inplace=True
    )
    mpdm = Mpo.onsite(mol_list, "a^\dagger", mol_idx_set={0}).apply(mpdm).normalize(1.0)

    # do the evolution
    # nsteps = 90  # too many steps, may take hours to finish
    nsteps = 40
    dt = 10.0

    occ = [mpdm.e_occupations]
    for i in range(nsteps):
        mpdm = mpdm.evolve(mpo, dt)
        occ.append(mpdm.e_occupations)
    # make it compatible with std data
    occ = np.array(occ[:nsteps]).transpose()

    with open(os.path.join(cur_dir, "FT_occ" + str(n_dmrg_phs) + ".npy"), "rb") as f:
        std = np.load(f)
    assert np.allclose(occ[:, :nsteps], std[:, :nsteps], atol=1e-3)
