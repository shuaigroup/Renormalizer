# -*- coding: utf-8 -*-

import pytest
import numpy as np

from ephMPS.model import Phonon, Mol
from ephMPS.sbm import SpinBosonModel, param2mollist, SpectralDensityFunction
from ephMPS.utils import Quantity, CompressConfig, EvolveConfig
from ephMPS.mps.tests.test_sbm import get_exact_zt


def test_sdf():
    alpha = 0.05
    omega_c = Quantity(5)
    sdf = SpectralDensityFunction(alpha, omega_c)
    omega_list, displacement_list = sdf.trapz(200, 0.0, 50)

    ph_list = [Phonon.simplest_phonon(o, d) for o,d in zip(omega_list, displacement_list)]
    mol = Mol(Quantity(0), ph_list, tunnel=Quantity(1))
    mol_reor = mol.reorganization_energy

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

    mol_list = param2mollist(alpha, raw_delta, raw_omega_c, 5, n_phonons)

    compress_config = CompressConfig(threshold=1e-4)
    evolve_config = EvolveConfig(adaptive=True, evolve_dt=0.1)
    sbm = SpinBosonModel(mol_list, Quantity(0), compress_config=compress_config, evolve_config=evolve_config)
    sbm.evolve(evolve_time=20)
    spin1 = sbm.sigma_z
    spin2 = get_exact_zt(mol_list[0], sbm.evolve_times)
    assert np.allclose(spin1, spin2, atol=1e-3)