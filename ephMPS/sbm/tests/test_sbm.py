# -*- coding: utf-8 -*-

import pytest
import numpy as np

from ephMPS.sbm import SpectralDensityFunction, SBM
from ephMPS.model import Phonon, Mol, MolList
from ephMPS.utils import Quantity, CompressConfig, EvolveConfig
from ephMPS.mps.tests.test_sbm import get_exact_zt


# todo: test for SpectralDensityFunction

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
    sdf = SpectralDensityFunction(alpha, raw_omega_c)

    delta, omega_c = sdf.adiabatic_renormalization(raw_delta, 5)

    omega_list, displacement_list = sdf.trapz(n_phonons, 0.0, omega_c.as_au())

    ph_list = [Phonon.simplest_phonon(o, d) for o,d in zip(omega_list, displacement_list)]
    mol = Mol(Quantity(0), ph_list, tunnel=delta)
    mol_list = MolList([mol], None)

    compress_config = CompressConfig(threshold=1e-4)
    evolve_config = EvolveConfig(adaptive=True, evolve_dt=0.1)
    sbm = SBM(mol_list, Quantity(0), compress_config=compress_config, evolve_config=evolve_config)
    sbm.evolve(evolve_time=20)
    spin1 = sbm.spin
    spin2 = get_exact_zt(mol, sbm.evolve_times)
    assert np.allclose(spin1, spin2, atol=1e-3)


