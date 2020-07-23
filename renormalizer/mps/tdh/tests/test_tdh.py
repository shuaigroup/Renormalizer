# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import logging
import os

import numpy as np
import pytest

from renormalizer.model import HolsteinModel, Mol
from renormalizer.mps import Mpo
from renormalizer.mps.gs import construct_mps_mpo_2, optimize_mps
from renormalizer.mps.tdh import tdh
from renormalizer.tests.parameter import (
    hartree_holstein_model,
    ph_phys_dim,
    custom_model,
    hartree_ph_list,
    elocalex,
    dipole_abs,
)
from renormalizer.mps.tdh.tests import cur_dir
from renormalizer.utils import Quantity, log, constant


def test_SCF():

    #  EX
    nexciton = 1
    WFN, Etot = tdh.SCF(hartree_holstein_model, nexciton)
    assert Etot == pytest.approx(0.0843103276663)

    fe, fv = 1, 6
    HAM, Etot, A_el = tdh.construct_H_Ham(
        hartree_holstein_model, nexciton, WFN, fe, fv, debug=True
    )
    assert Etot == pytest.approx(0.0843103276663)
    occ_std = np.array([[0.20196397], [0.35322702], [0.444809]])
    assert np.allclose(A_el, occ_std)

    # GS
    nexciton = 0
    WFN, Etot = tdh.SCF(hartree_holstein_model, nexciton)
    assert Etot == pytest.approx(0.0)

    fe, fv = 1, 6
    HAM, Etot, A_el = tdh.construct_H_Ham(
        hartree_holstein_model, nexciton, WFN, fe, fv, debug=True
    )
    assert Etot == pytest.approx(0.0)
    occ_std = np.array([[0.0], [0.0], [0.0]])
    assert np.allclose(A_el, occ_std)


def test_SCF_exact():

    nexciton = 1

    # e-ph coupling constant is set to zero
    dmrg_model = custom_model(None, ph_phys_dim, dis=[Quantity(0), Quantity(0)])
    # DMRG calculation
    procedure = [[40, 0.4], [40, 0.2], [40, 0.1], [40, 0], [40, 0]]
    mps, mpo = construct_mps_mpo_2(dmrg_model, 40, nexciton)
    mps.optimize_config.procedure = procedure
    energies, _ = optimize_mps(mps, mpo)
    dmrg_e = energies[-1]

    # print occupation
    dmrg_occ = []
    for i in [0, 1, 2]:
        mpo = Mpo.onsite(dmrg_model, r"a^\dagger a", dipole=False, mol_idx_set={i})
        dmrg_occ.append(mps.expectation(mpo))
    print("dmrg_occ", dmrg_occ)

    hartree_model = custom_model(
        None, ph_phys_dim, dis=[Quantity(0), Quantity(0)], hartrees=[True, True]
    )
    WFN, Etot = tdh.SCF(hartree_model, nexciton)
    assert Etot + dmrg_model.gs_zpe == pytest.approx(dmrg_e)

    fe, fv = 1, 6
    HAM, Etot, A_el = tdh.construct_H_Ham(
        hartree_model, nexciton, WFN, fe, fv, debug=True
    )
    assert Etot + dmrg_model.gs_zpe == pytest.approx(dmrg_e)
    assert np.allclose(A_el.flatten(), dmrg_occ, rtol=1e-4)


def test_TDH_ZT_emi():

    # disable tons of logging information
    log.init_log(logging.WARNING)
    nsteps = 3000 - 1
    dt = 30.0

    ls = tdh.LinearSpectra("emi", hartree_holstein_model, prop_method="unitary")
    ls.evolve(dt, nsteps)
    with open(os.path.join(cur_dir, "TDH_ZT_emi_prop1.npy"), "rb") as f:
        TDH_ZT_emi_prop1_std = np.load(f)
    assert np.allclose(ls.autocorr, TDH_ZT_emi_prop1_std)

    ls = tdh.LinearSpectra("emi", hartree_holstein_model, prop_method="C_RK4")
    ls.evolve(dt, nsteps)
    with open(os.path.join(cur_dir, "TDH_ZT_emi_RK4.npy"), "rb") as f:
        TDH_ZT_emi_RK4_std = np.load(f)
    assert np.allclose(ls.autocorr, TDH_ZT_emi_RK4_std, rtol=1e-2)


def test_TDH_ZT_abs():

    log.init_log(logging.WARNING)
    nsteps = 300 - 1
    dt = 10.0

    E_offset = -2.28614053 / constant.au2ev

    ls = tdh.LinearSpectra(
        "abs", hartree_holstein_model, E_offset=E_offset, prop_method="unitary"
    )
    ls.evolve(dt, nsteps)
    with open(os.path.join(cur_dir, "TDH_ZT_abs_prop1.npy"), "rb") as f:
        TDH_ZT_abs_prop1_std = np.load(f)
    assert np.allclose(ls.autocorr, TDH_ZT_abs_prop1_std)

    ls = tdh.LinearSpectra(
        "abs", hartree_holstein_model, E_offset=E_offset, prop_method="C_RK4"
    )
    ls.evolve(dt, nsteps)
    with open(os.path.join(cur_dir, "TDH_ZT_abs_RK4.npy"), "rb") as f:
        TDH_ZT_abs_RK4_std = np.load(f)
    assert np.allclose(ls.autocorr, TDH_ZT_abs_RK4_std, rtol=1e-3)


def test_1mol_ZTabs():
    log.init_log(logging.WARNING)
    nmols = 1

    model = HolsteinModel([Mol(elocalex, hartree_ph_list, dipole_abs) for _ in range(nmols)], np.zeros([1, 1]), )

    E_offset = -model[0].elocalex - model[0].e0

    ls = tdh.LinearSpectra("abs", model, E_offset=E_offset, prop_method="unitary")

    nsteps = 1000 - 1
    dt = 30.0
    ls.evolve(dt, nsteps)

    with open(os.path.join(cur_dir, "1mol_ZTabs.npy"), "rb") as f:
        mol1_ZTabs_std = np.load(f)

    assert np.allclose(ls.autocorr, mol1_ZTabs_std)
