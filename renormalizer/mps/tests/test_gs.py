# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import os

import numpy as np
import pytest

from renormalizer.model import Model, h_qc
from renormalizer.mps.backend import primme
from renormalizer.mps.gs import construct_mps_mpo, optimize_mps, DmrgFCISolver
from renormalizer.mps import Mpo, Mps, StackedMpo
from renormalizer.tests.parameter import holstein_model
from renormalizer.utils.configs import OFS
from renormalizer.mps.tests import cur_dir


nexciton = 1
procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]

GS_E = 0.08401412 + holstein_model.gs_zpe

@pytest.mark.parametrize("scheme", (
        1,
        4,
))
@pytest.mark.parametrize("method", (
        "1site",
        "2site",
))
def test_optimization(scheme, method):
    mps, mpo = construct_mps_mpo(holstein_model.switch_scheme(scheme), procedure[0][0], nexciton)
    mps.optimize_config.procedure = procedure
    mps.optimize_config.method = method
    energies, mps_opt = optimize_mps(mps.copy(), mpo)
    assert energies[-1] == pytest.approx(GS_E, rel=1e-5)
    assert mps_opt.expectation(mpo) == pytest.approx(GS_E, rel=1e-5)


@pytest.mark.parametrize("method", (
        "1site",
        "2site",
))
@pytest.mark.parametrize("algo", (
        "davidson",
        pytest.param("primme", marks=pytest.mark.skipif(primme is None, reason="primme not installed"))
))
def test_multistate(method, algo):
    mps, mpo = construct_mps_mpo(holstein_model, procedure[0][0], nexciton)
    mps.optimize_config.procedure = procedure
    mps.optimize_config.nroots = 4
    mps.optimize_config.method = method
    mps.optimize_config.algo = algo
    mps.optimize_config.e_atol = 1e-6
    mps.optimize_config.e_rtol = 1e-6
    energy, mps = optimize_mps(mps, mpo)
    expectation = [mp.expectation(mpo) for mp in mps]
    energy_std = np.array([0.08401412, 0.08449771, 0.08449801, 0.08449945]) + holstein_model.gs_zpe
    assert np.allclose(energy[-1], energy_std)
    assert np.allclose(expectation, energy_std)


@pytest.mark.parametrize("method", (
        "1site",
        "2site",
))
@pytest.mark.parametrize("nroots", (
        1,
        4,
))
def test_ex(method, nroots):
    mps, mpo = construct_mps_mpo(holstein_model, procedure[0][0], nexciton)
    mps.optimize_config.procedure = procedure
    mps.optimize_config.nroots = nroots
    mps.optimize_config.method = method
    mps.optimize_config.e_atol = 1e-6
    mps.optimize_config.e_rtol = 1e-6
    omega = 0.084
    energy, mps = optimize_mps(mps, mpo, omega=omega)
    energy_std = np.array([0.08401412, 0.08449771, 0.08449801, 0.08449945]) + holstein_model.gs_zpe
    if nroots == 1:
        #print("eigenenergy", mps.expectation(mpo))
        assert np.allclose(mps.expectation(mpo), energy_std[0])
    else:
        #print("eigenenergy", [ms.expectation(mpo) for ms in mps])
        assert np.allclose([ms.expectation(mpo) for ms in mps], energy_std)


def test_ofs():
    # `switch_scheme` makes copy, so `holstein_model` is not changed during OFS
    mps, mpo = construct_mps_mpo(holstein_model.switch_scheme(1), procedure[0][0], nexciton)
    # transform from HolsteinModel to the general Model
    mps.model = Model(mps.model.basis, mps.model.ham_terms)
    mps.optimize_config.procedure = procedure
    mps.optimize_config.method = "2site"
    mps.compress_config.ofs = OFS.ofs_s
    energies, mps_opt = optimize_mps(mps.copy(), mpo)
    assert energies[-1] == pytest.approx(GS_E, rel=1e-5)
    mpo = Mpo(mps_opt.model)
    assert mps_opt.expectation(mpo) == pytest.approx(GS_E, rel=1e-5)


@pytest.mark.parametrize("with_ofs", (True, False))
@pytest.mark.parametrize("stacked", (True, False))
def test_qc(with_ofs, stacked):
    """
    m = M(atom=[["H", np.cos(theta), np.sin(theta), 0] for theta in 2*np.pi/6 * np.arange(6)], basis="STO-3G")
    hf = m.HF()
    hf.kernel()
    fcidump.from_mo(m, "H6.txt", hf.mo_coeff)
    hf.CASCI(ncas=6, nelecas=6).kernel()
    """
    spatial_norbs = 6
    h1e, h2e, nuc = h_qc.read_fcidump(os.path.join(cur_dir, "H6.txt"), spatial_norbs)

    basis, ham_terms = h_qc.qc_model(h1e, h2e, stacked=stacked)

    if stacked:
        stacked_mpo = []
        for i in range(len(ham_terms)):
            model = Model(basis, ham_terms[i])
            impo = Mpo(model)
            stacked_mpo.append(impo)
        mpo = StackedMpo(stacked_mpo)
    else:
        model = Model(basis, ham_terms)
        mpo = Mpo(model)

    fci_e = -3.23747673055271 - nuc

    nelec = [3, 3]
    M = 30
    procedure = [[M, 0.4], [M, 0.2], [M, 0.1], [M, 0], [M, 0], [M, 0], [M, 0]]
    np.random.seed(2023)
    mps = Mps.random(model, nelec, M, percent=1.0)
    hf = Mps.hartree_product_state(model, {i:1 for i in range(sum(nelec))})
    mps = mps.scale(1e-8)+hf
    #print("hf energy", mps.expectation(mpo))
    mps.optimize_config.procedure = procedure
    mps.optimize_config.method = "2site"
    if with_ofs:
        mps.compress_config.ofs = OFS.ofs_s
        mps.compress_config.ofs_swap_jw = True
    energies, mps = optimize_mps(mps.copy(), mpo)
    print(mpo)
    gs_e = min(energies)
    assert np.allclose(gs_e, fci_e, atol=5e-3)


def test_stackedmpo():
    scheme = 1
    method = '1site'
    mps, mpo = construct_mps_mpo(holstein_model.switch_scheme(scheme), procedure[0][0], nexciton)
    mps.optimize_config.procedure = procedure
    mps.optimize_config.method = method
    energies1, _ = optimize_mps(mps.copy(), mpo)
    energies2, _ = optimize_mps(mps.copy(), StackedMpo([mpo, mpo]))
    assert np.all(np.abs(np.array(energies2) - np.array(energies1) * 2) < 1e-8)


def test_pyscf_solver():
    try:
        from pyscf import M, mcscf, fci
    except ImportError:
        pytest.skip("pyscf not installed")
        return # make IDE happy

    # the unused code is for debugging
    # the test might not pass based on the unused code
    if True:
        mol = M(atom=[["N", 0, 0, 0], ["N", 0, 0, 1]], basis="sto3g")
        active_space = (10, 8)
        active_space = (6, 6)
    else:
        mol = M(atom=[["Li", 0, 0, 0], ["H", 0, 0, 1]], basis="sto3g")
        active_space = (2, 5)
    hf = mol.HF()
    hf.kernel()

    casci = mcscf.CASCI(hf, active_space[1], active_space[0])
    casci.kernel()
    e_ref = casci.e_tot
    rdm1_ref = casci.make_rdm1()
    rdm2_ref = fci.direct_spin1.make_rdm12(casci.ci, casci.ncas, casci.nelecas)[1]

    casci = mcscf.CASCI(hf, active_space[1], active_space[0])
    casci.fcisolver = DmrgFCISolver()
    casci.kernel()
    rdm1 = casci.make_rdm1()
    rdm2 = casci.fcisolver.make_rdm2(casci.fcisolver.mps, casci.ncas, casci.nelecas)

    # reasonable error since the bond dimension is only 32
    assert casci.e_tot == pytest.approx(e_ref, abs=1e-2)
    np.testing.assert_allclose(rdm1, rdm1_ref, atol=1e-2)
    np.testing.assert_allclose(rdm2, rdm2_ref, atol=1e-2)
