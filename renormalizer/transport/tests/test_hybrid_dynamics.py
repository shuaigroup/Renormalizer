# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import os

import numpy as np
import pytest

from renormalizer.mps import Mps, Mpo, MpDm, ThermalProp
from renormalizer.utils import Quantity, CompressConfig, EvolveMethod, EvolveConfig
from renormalizer.tests import parameter_PBI
from renormalizer.transport.tests import cur_dir


@pytest.mark.parametrize("scheme", (4, 3))
@pytest.mark.parametrize("n_dmrg_phs", (10, 5))
def test_zt(n_dmrg_phs, scheme):

    mol_list = parameter_PBI.construct_mol(4, n_dmrg_phs, 10 - n_dmrg_phs).switch_scheme(scheme)
    mps = Mps.ground_state(mol_list, False)
    # create electron
    mps = Mpo.onsite(mol_list, r"a^\dagger", mol_idx_set={0}).apply(mps).normalize(1.0)
    tentative_mpo = Mpo(mol_list)
    offset = mps.expectation(tentative_mpo)
    mpo = Mpo(mol_list, offset=Quantity(offset, "a.u."))
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
    assert np.allclose(occ, std, rtol=1e-2, atol=1e-4)


@pytest.mark.parametrize("scheme", (4, 3))
@pytest.mark.parametrize("n_dmrg_phs", (10, 5))
def test_FT_dynamics_hybrid_TDDMRG_TDH(n_dmrg_phs, scheme):

    mol_list = parameter_PBI.construct_mol(4, n_dmrg_phs, 10 - n_dmrg_phs).switch_scheme(scheme)
    mpdm = MpDm.max_entangled_gs(mol_list)
    tentative_mpo = Mpo(mol_list)
    temperature = Quantity(2000, "K")
    tp = ThermalProp(mpdm, tentative_mpo, exact=True, space="GS")
    tp.evolve(None, 1, temperature.to_beta() / 2j)
    mpdm = (
        Mpo.onsite(mol_list, r"a^\dagger", mol_idx_set={0}).apply(tp.latest_mps).normalize(1.0)
    )
    mpdm.compress_config = CompressConfig(max_bonddim=12)
    mpdm.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    offset = mpdm.expectation(tentative_mpo)
    mpo = Mpo(mol_list, offset=Quantity(offset, "a.u."))
    mpdm = mpdm.expand_bond_dimension(mpo)

    # do the evolution
    # nsteps = 90  # too many steps, may take hours to finish
    nsteps = 20
    dt = 20.0

    occ = [mpdm.e_occupations]
    for i in range(nsteps):
        mpdm = mpdm.evolve(mpo, dt)
        occ.append(mpdm.e_occupations)
    # make it compatible with std data
    occ = np.array(occ[:nsteps]).transpose()

    with open(os.path.join(cur_dir, "FT_occ" + str(n_dmrg_phs) + ".npy"), "rb") as f:
        std = np.load(f)
    assert np.allclose(occ[:, :nsteps], std[:, :2*nsteps:2], atol=1e-3, rtol=1e-3)
