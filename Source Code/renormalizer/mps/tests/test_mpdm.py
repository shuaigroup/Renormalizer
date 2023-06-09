# -*- coding: utf-8 -*-

import numpy as np
import pytest

from renormalizer.mps import Mps, Mpo, MpDm, MpDmFull, ThermalProp
from renormalizer.model import HolsteinModel, Mol, Phonon
from renormalizer.tests import parameter
from renormalizer.utils import Quantity, EvolveConfig, EvolveMethod


@pytest.mark.parametrize("nmols", (2, 3, 4))
@pytest.mark.parametrize("phonon_freq", (0.01, 0.001, 0.0001))
def test_mpdm_full(nmols, phonon_freq):
    ph = Phonon.simple_phonon(Quantity(phonon_freq), Quantity(1), 2)
    m = Mol(Quantity(0), [ph])
    model = HolsteinModel([m] * nmols, Quantity(1), )

    gs_dm = MpDm.max_entangled_gs(model)
    beta = Quantity(1000, "K").to_beta()
    tp = ThermalProp(gs_dm, Mpo(model), exact=True, space="GS")
    tp.evolve(None, 50, beta / 1j)
    gs_dm = tp.latest_mps
    assert np.allclose(gs_dm.e_occupations, [0] * nmols)
    e_gs_dm = Mpo.onsite(model, r"a^\dagger", dof_set={0}).apply(gs_dm, canonicalise=True)
    assert np.allclose(e_gs_dm.e_occupations, [1] + [0] * (nmols - 1))

    mpdm_full = MpDmFull.from_mpdm(e_gs_dm)
    assert np.allclose(mpdm_full.e_occupations, e_gs_dm.e_occupations)
    assert np.allclose(mpdm_full.ph_occupations, e_gs_dm.ph_occupations, rtol=1e-3)


def test_from_mps():
    gs = Mps.random(parameter.holstein_model, 1, 20)
    gs_mpdm = MpDm.from_mps(gs)
    assert np.allclose(gs.e_occupations, gs_mpdm.e_occupations)
    gs = gs.canonicalise()
    gs_mpdm = gs_mpdm.canonicalise()
    assert np.allclose(gs.e_occupations, gs_mpdm.e_occupations)


@pytest.mark.parametrize(
    "adaptive, evolve_method",
    (
        [True,  EvolveMethod.tdvp_ps],
        [False, EvolveMethod.prop_and_compress],
        [False, EvolveMethod.tdvp_mu_vmf],
    ),
)
def test_thermal_prop(adaptive, evolve_method):
    model = parameter.holstein_model
    init_mps = MpDm.max_entangled_ex(model)
    mpo = Mpo(model)
    beta = Quantity(298, "K").to_beta()
    evolve_time = beta / 2j

    evolve_config = EvolveConfig(evolve_method, adaptive=adaptive, guess_dt=0.1/1j)

    if adaptive:
        nsteps = 1
    else:
        nsteps = 100

    if evolve_method == EvolveMethod.tdvp_mu_vmf:
        nsteps = 20
        evolve_config.ivp_rtol = 1e-3
        evolve_config.ivp_atol = 1e-6
        evolve_config.reg_epsilon = 1e-8
        init_mps.compress_config.bond_dim_max_value = 12

    dbeta = evolve_time/nsteps

    tp = ThermalProp(init_mps, mpo, evolve_config=evolve_config)
    tp.evolve(evolve_dt=dbeta, nsteps=nsteps)
    # MPO, HAM, Etot, A_el = mps.construct_hybrid_Ham(mpo, debug=True)
    # exact A_el: 0.20896541050347484, 0.35240029674394463, 0.4386342927525734
    # exact internal energy: 0.0853388060014744
    etot_std = 0.0853388 + parameter.holstein_model.gs_zpe
    occ_std = [0.20896541050347484, 0.35240029674394463, 0.4386342927525734]
    rtol = 5e-3
    assert np.allclose(tp.e_occupations_array[-1], occ_std, rtol=rtol)
    assert np.allclose(tp.energies[-1], etot_std, rtol=rtol)

