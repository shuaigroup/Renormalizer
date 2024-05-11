# -*- coding: utf-8 -*-

import pytest

from renormalizer.model import Phonon
from renormalizer.sbm import SpinBosonDynamics, param2mollist, SpectralDensityFunction
from renormalizer.utils import Quantity, EvolveConfig, EvolveMethod
from renormalizer.mps.tests.test_sbm import get_qutip_zt

import numpy as np

def test_sdf():
    alpha = 0.05
    omega_c = Quantity(5)
    sdf = SpectralDensityFunction(alpha, omega_c)
    omega_list, displacement_list = sdf.trapz(200, 0.0, 50)
    omega_list, displacement_list = sdf.post_process(omega_list, displacement_list)

    ph_list = [Phonon.simplest_phonon(o, d) for o,d in zip(omega_list, displacement_list)]
    mol_reor = sum(ph.reorganization_energy.as_au() for ph in ph_list)

    assert mol_reor == pytest.approx(alpha * omega_c.as_au() / 2, abs=0.005)


@pytest.mark.parametrize(
    "alpha",
    (
        0.05,
        0.5
    ),
)
def test_sbm_zt(alpha):
    raw_delta = Quantity(1)
    raw_omega_c = Quantity(20)
    n_phonons = 3

    model = param2mollist(alpha, raw_delta, raw_omega_c, 5, n_phonons)

    evolve_config = EvolveConfig(method=EvolveMethod.tdvp_ps, adaptive=True, guess_dt=0.1)
    sbm = SpinBosonDynamics(model, Quantity(0), evolve_config=evolve_config)
    sbm.evolve(nsteps=20, evolve_time=20)
    spin1 = sbm.sigma_z
    spin2 = get_qutip_zt(model, sbm.evolve_times)
    assert np.allclose(spin1, spin2, atol=1e-3)
