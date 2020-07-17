# -*- coding: utf-8 -*-

import numpy as np
import qutip
import pytest

from renormalizer.model import Phonon, Mol, SpinBosonModel
from renormalizer.mps import Mps, Mpo, MpDm, ThermalProp
from renormalizer.utils import Quantity, CompressConfig, EvolveConfig


def get_model():
    nphonons = 5
    ph_levels = 2

    epsilon = 1
    delta = 1

    ph_list = [Phonon.simple_phonon(Quantity(1), Quantity(1), ph_levels)] * nphonons
    return SpinBosonModel(Quantity(epsilon), Quantity(delta), ph_list)

def test_zt():
    mol_list = get_model()

    mps = Mps.ground_state(mol_list, False)
    mps.compress_config = CompressConfig(threshold=1e-6)
    mps.evolve_config = EvolveConfig(adaptive=True, guess_dt=0.1)
    mps.use_dummy_qn = True
    mpo = Mpo(mol_list)
    time_series = [0]
    spin = [1]
    for i in range(30):
        dt = mps.evolve_config.guess_dt
        mps = mps.evolve(mpo, evolve_dt=dt)
        time_series.append(time_series[-1] + dt)
        spin.append(mps.expectation(Mpo.onsite(mol_list, "sigma_z")))
    qutip_res = get_qutip_zt(mol_list, time_series)
    assert np.allclose(qutip_res, spin, atol=1e-3)


@pytest.mark.xfail(reason="thermal prop calculate electron occupations (not valid for spin)")
def test_ft():
    mol_list = get_model()
    mpo = Mpo(mol_list)
    impdm = MpDm.max_entangled_gs(mol_list)
    impdm.compress_config = CompressConfig(threshold=1e-6)
    impdm.use_dummy_qn = True
    temperature = Quantity(3)
    evolve_config = EvolveConfig(adaptive=True, guess_dt=-0.001j)
    tp = ThermalProp(impdm, mpo, evolve_config=evolve_config)
    tp.evolve(nsteps=1, evolve_time=temperature.to_beta() / 2j)
    mpdm = tp.latest_mps
    mpdm = Mpo.onsite(mol_list, r"sigma_x").contract(mpdm)
    mpdm.evolve_config = EvolveConfig(adaptive=True, guess_dt=0.1)
    time_series = [0]
    spin = [1 - 2 * mpdm.e_occupations[0]]
    for i in range(30):
        dt = mpdm.evolve_config.guess_dt
        mpdm = mpdm.evolve(mpo, evolve_dt=dt)
        time_series.append(time_series[-1] + dt)
        spin.append(mpdm.expectation(Mpo.onsite(mol_list, "sigma_z")))
    qutip_res = get_qutip_ft(mol_list, temperature, time_series)
    assert np.allclose(qutip_res, spin, atol=1e-3)


def get_qutip_operator(mol_list):
    blist = []
    for i, ph1 in enumerate(mol_list.ph_list):
        basis = [qutip.identity(2)]
        for j, ph2 in enumerate(mol_list.ph_list):
            if j == i:
                state = qutip.destroy(ph1.n_phys_dim)
            else:
                state = qutip.identity(ph2.n_phys_dim)
            basis.append(state)
        blist.append(qutip.tensor(basis))

    ph_iden = [qutip.identity(ph.n_phys_dim) for ph in mol_list.ph_list]

    sigma_x = qutip.tensor([qutip.sigmax()] + ph_iden)
    sigma_z = qutip.tensor([qutip.sigmaz()] + ph_iden)
    terms = [mol_list.delta * sigma_x, mol_list.epsilon * sigma_z]
    for i, ph in enumerate(mol_list.ph_list):
        g = ph.coupling_constant
        terms.append(ph.omega[0] * blist[i].dag() * blist[i])
        terms.append(ph.omega[0] * g * sigma_z * (blist[i].dag() + blist[i]))
    H = sum(terms)

    return H, sigma_x, sigma_z


def get_qutip_zt(mol_list, time_series):
    H, _, sigma_z = get_qutip_operator(mol_list)
    init_state = qutip.tensor([qutip.basis(2)] + [qutip.basis(ph.n_phys_dim) for ph in mol_list.ph_list])
    result = qutip.mesolve(H, init_state, time_series, e_ops=[sigma_z])
    return result.expect[0]


def get_qutip_ft(mol_list, temperature, time_series):
    H, sigma_x, sigma_z = get_qutip_operator(mol_list)
    init_state =  sigma_x * (-temperature.to_beta() * H).expm().unit() * sigma_x.dag()
    result = qutip.mesolve(H, init_state, time_series, e_ops=[sigma_z])
    return result.expect[0]
