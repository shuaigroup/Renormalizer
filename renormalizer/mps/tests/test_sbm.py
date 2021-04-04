# -*- coding: utf-8 -*-

import numpy as np
import qutip

from renormalizer.model import Phonon, SpinBosonModel
from renormalizer.mps import Mps, Mpo, MpDm, ThermalProp
from renormalizer.utils import Quantity, CompressConfig, EvolveConfig
from renormalizer.model.op import Op


def get_model():
    nphonons = 5
    ph_levels = 2

    epsilon = 1
    delta = 1

    ph_list = [Phonon.simple_phonon(Quantity(1), Quantity(1), ph_levels)] * nphonons
    return SpinBosonModel(Quantity(epsilon), Quantity(delta), ph_list)

def test_zt():
    model = get_model()

    mps = Mps.ground_state(model, False)
    mps.compress_config = CompressConfig(threshold=1e-6)
    mps.evolve_config = EvolveConfig(adaptive=True, guess_dt=0.1)
    mps.use_dummy_qn = True
    mpo = Mpo(model)
    time_series = [0]
    spin = [1]
    sigma_z_oper = Mpo(model, Op("sigma_z", "spin"))
    for i in range(30):
        dt = mps.evolve_config.guess_dt
        mps = mps.evolve(mpo, evolve_dt=dt)
        time_series.append(time_series[-1] + dt)
        spin.append(mps.expectation(sigma_z_oper))
    qutip_res = get_qutip_zt(model, time_series)
    assert np.allclose(qutip_res, spin, atol=1e-3)


def test_ft():
    model = get_model()
    mpo = Mpo(model)
    impdm = MpDm.max_entangled_gs(model)
    impdm.compress_config = CompressConfig(threshold=1e-6)
    temperature = Quantity(3)
    evolve_config = EvolveConfig(adaptive=True, guess_dt=-0.001j)
    tp = ThermalProp(impdm, evolve_config=evolve_config)
    tp.evolve(nsteps=1, evolve_time=temperature.to_beta() / 2j)
    mpdm = tp.latest_mps
    mpdm = Mpo(model, Op("sigma_x", "spin")).contract(mpdm)
    mpdm.evolve_config = EvolveConfig(adaptive=True, guess_dt=0.1)
    time_series = [0]
    sigma_z_oper = Mpo(model, Op("sigma_z", "spin"))
    spin = [mpdm.expectation(sigma_z_oper)]
    for i in range(29):
        dt = mpdm.evolve_config.guess_dt
        mpdm = mpdm.evolve(mpo, evolve_dt=dt)
        time_series.append(time_series[-1] + dt)
        spin.append(mpdm.expectation(sigma_z_oper))
    qutip_res = get_qutip_ft(model, temperature, time_series)
    assert np.allclose(qutip_res, spin, atol=1e-3)


def get_qutip_operator(model):
    blist = []
    for i, ph1 in enumerate(model.ph_list):
        basis = [qutip.identity(2)]
        for j, ph2 in enumerate(model.ph_list):
            if j == i:
                state = qutip.destroy(ph1.n_phys_dim)
            else:
                state = qutip.identity(ph2.n_phys_dim)
            basis.append(state)
        blist.append(qutip.tensor(basis))

    ph_iden = [qutip.identity(ph.n_phys_dim) for ph in model.ph_list]

    sigma_x = qutip.tensor([qutip.sigmax()] + ph_iden)
    sigma_z = qutip.tensor([qutip.sigmaz()] + ph_iden)
    terms = [model.delta * sigma_x, model.epsilon * sigma_z]
    for i, ph in enumerate(model.ph_list):
        g = ph.coupling_constant
        terms.append(ph.omega[0] * blist[i].dag() * blist[i])
        terms.append(ph.omega[0] * g * sigma_z * (blist[i].dag() + blist[i]))
    H = sum(terms)

    return H, sigma_x, sigma_z


def get_qutip_zt(model, time_series):
    H, _, sigma_z = get_qutip_operator(model)
    init_state = qutip.tensor([qutip.basis(2)] + [qutip.basis(ph.n_phys_dim) for ph in model.ph_list])
    result = qutip.mesolve(H, init_state, time_series, e_ops=[sigma_z])
    return result.expect[0]


def get_qutip_ft(model, temperature, time_series):
    H, sigma_x, sigma_z = get_qutip_operator(model)
    init_state =  sigma_x * (-temperature.to_beta() * H).expm().unit() * sigma_x.dag()
    result = qutip.mesolve(H, init_state, time_series, e_ops=[sigma_z])
    return result.expect[0]
