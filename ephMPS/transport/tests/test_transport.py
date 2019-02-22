# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

import numpy as np
import pytest

from ephMPS.model import Phonon, Mol, MolList
from ephMPS.transport import ChargeTransport, EDGE_THRESHOLD
from ephMPS.transport.transport import (
    calc_reduced_density_matrix,
    calc_reduced_density_matrix_straight,
)
from ephMPS.utils import Quantity
from ephMPS.utils import EvolveMethod, EvolveConfig, RungeKutta

# the temperature should be compatible with the low vibration frequency in TestBandLimitFiniteT
# otherwise underflow happens in exact propagator
low_t = Quantity(1e-7, "K")

mol_num = 13
ph_list = [
    Phonon.simple_phonon(
        Quantity(omega, "cm^{-1}"), Quantity(displacement, "a.u."), 4
    )
    for omega, displacement in [[1e-10, 1e-10]]
]
j_constant = Quantity(0.8, "eV")
band_limit_mol_list = MolList(
    [Mol(Quantity(3.87e-3, "a.u."), ph_list)] * mol_num, j_constant
)

def get_analytical_r_square(time_series):
    return  2 * (j_constant.as_au()) ** 2 * time_series ** 2

def assert_band_limit(ct, rtol):
    analytical_r_square = get_analytical_r_square(ct.evolve_times_array)
    # has evolved to the edge
    assert EDGE_THRESHOLD < ct.latest_mps.e_occupations[0]
    # value OK
    assert np.allclose(analytical_r_square, ct.r_square_array, rtol=rtol)

@pytest.mark.parametrize(
    "method, evolve_dt, nsteps, rtol",
    (
            (EvolveMethod.prop_and_compress, 4, 25, 1e-3),
            # not working. Moves slightly slower. Dunno why.
            #(EvolveMethod.tdvp_mctdh_new, 0.5, 200, 1e-2),
            (EvolveMethod.tdvp_ps, 2, 50, 1e-3),
    )
)
def test_bandlimit_zero_t(method, evolve_dt, nsteps, rtol):
    evolve_config = EvolveConfig(method)
    if method != EvolveMethod.prop_and_compress:
        evolve_config.expected_bond_order = 10
    ct = ChargeTransport(band_limit_mol_list, evolve_config=evolve_config)
    ct.stop_at_edge = True
    ct.evolve(evolve_dt, nsteps)
    assert_band_limit(ct, rtol)


# from matplotlib import pyplot as plt
# plt.plot(ct.r_square_array)
# plt.plot(analytical_r_square)
# plt.show()

@pytest.mark.parametrize("init_dt", (1e-1, 20))
def test_adaptive_zero_t(init_dt):
    rk_config = RungeKutta("RKF45", evolve_dt=init_dt)
    evolve_config = EvolveConfig(rk_config=rk_config)
    ct = ChargeTransport(band_limit_mol_list, evolve_config=evolve_config)
    ct.stop_at_edge = True
    ct.evolve()
    assert_band_limit(ct, 1e-2)

@pytest.mark.parametrize(
    "mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps",
    (
        [5, 0.8, 3.87e-3, [[1345.6738910804488, 16.274571056529368]], 4, 2, 25],
        [5, 0.8, 3.87e-3, [[1e-10, 1e-10]], 4, 2, 25],
    ),
)
def test_economic_mode(
    mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps
):
    ph_list = [
        Phonon.simple_phonon(
            Quantity(omega, "cm^{-1}"), Quantity(displacement, "a.u."), ph_phys_dim
        )
        for omega, displacement in ph_info
    ]
    mol_list = MolList(
        [Mol(Quantity(elocalex_value, "a.u."), ph_list)] * mol_num,
        Quantity(j_constant_value, "eV"),
    )
    ct1 = ChargeTransport(mol_list)
    ct1.evolve(evolve_dt, nsteps)
    ct2 = ChargeTransport(mol_list)
    ct2.economic_mode = True
    ct2.evolve(evolve_dt, nsteps)
    assert ct1.is_similar(ct2)
    assert ct1.get_dump_dict() == ct2.get_dump_dict()


@pytest.mark.parametrize(
    "mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps",
    ([5, 0.8, 3.87e-3, [[1345.6738910804488, 16.274571056529368]], 4, 2, 25],),
)
def test_memory_limit(
    mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps
):
    ph_list = [
        Phonon.simple_phonon(
            Quantity(omega, "cm^{-1}"), Quantity(displacement, "a.u."), ph_phys_dim
        )
        for omega, displacement in ph_info
    ]
    mol_list = MolList(
        [Mol(Quantity(elocalex_value, "a.u."), ph_list)] * mol_num,
        Quantity(j_constant_value, "eV"),
    )
    ct1 = ChargeTransport(mol_list)
    ct1.evolve(evolve_dt, nsteps)
    ct2 = ChargeTransport(mol_list)
    ct2.memory_limit = 2 ** 20 / 4
    ct2.evolve(evolve_dt, nsteps)
    assert ct1.is_similar(ct2, rtol=1e-2)


@pytest.mark.parametrize(
    "mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps",
    ([5, 0.8, 3.87e-3, [[1345.6738910804488, 16.274571056529368]], 4, 2, 50],),
)
def test_compress_add(
    mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps
):
    ph_list = [
        Phonon.simple_phonon(
            Quantity(omega, "cm^{-1}"), Quantity(displacement, "a.u."), ph_phys_dim
        )
        for omega, displacement in ph_info
    ]
    mol_list = MolList(
        [Mol(Quantity(elocalex_value, "a.u."), ph_list)] * mol_num,
        Quantity(j_constant_value, "eV"),
    )
    ct1 = ChargeTransport(mol_list, temperature=Quantity(298, "K"))
    ct1.reduced_density_matrices = None
    ct1.set_threshold(1e-5)
    ct1.evolve(evolve_dt, nsteps)
    ct2 = ChargeTransport(mol_list, temperature=Quantity(298, "K"))
    ct2.reduced_density_matrices = None
    ct2.set_threshold(1e-5)
    ct2.latest_mps.compress_add = True
    ct2.evolve(evolve_dt, nsteps)
    assert ct1.is_similar(ct2, rtol=1e-2)


def check_rdm(mps):
    rdm1 = calc_reduced_density_matrix_straight(mps)
    rdm2 = calc_reduced_density_matrix(mps)
    assert np.allclose(rdm1, rdm2)


@pytest.mark.parametrize(
    "mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps, temperature",
    (
        [3, 0.8, 3.87e-3, [[1345.6738910804488, 16.274571056529368]], 4, 2, 15, 0],
        [3, 0.8, 3.87e-3, [[1345.6738910804488, 16.274571056529368]], 4, 2, 15, 100],
    ),
)
def test_reduced_density_matrix(
    mol_num,
    j_constant_value,
    elocalex_value,
    ph_info,
    ph_phys_dim,
    evolve_dt,
    nsteps,
    temperature,
):
    ph_list = [
        Phonon.simple_phonon(
            Quantity(omega, "cm^{-1}"), Quantity(displacement, "a.u."), ph_phys_dim
        )
        for omega, displacement in ph_info
    ]
    mol_list = MolList(
        [Mol(Quantity(elocalex_value, "a.u."), ph_list)] * mol_num,
        Quantity(j_constant_value, "eV"),
    )
    ct = ChargeTransport(mol_list, temperature=Quantity(temperature, "K"))
    check_rdm(ct.latest_mps)
    ct.evolve(evolve_dt, nsteps)
    for mps in ct.tdmps_list:
        check_rdm(mps)


@pytest.mark.parametrize(
    "mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps",
    ([5, 0.8, 3.87e-3, [[1400, 17]], 4, 2, 50],),
)
def test_similar(
    mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps
):
    ph_list = [
        Phonon.simple_phonon(
            Quantity(omega, "cm^{-1}"), Quantity(displacement, "a.u."), ph_phys_dim
        )
        for omega, displacement in ph_info
    ]
    mol_list = MolList(
        [Mol(Quantity(elocalex_value, "a.u."), ph_list)] * mol_num,
        Quantity(j_constant_value, "eV"),
    )
    ct1 = ChargeTransport(mol_list)
    ct1.evolve(evolve_dt, nsteps)
    ct2 = ChargeTransport(mol_list)
    ct2.evolve(evolve_dt + 1e-5, nsteps)
    assert ct1.is_similar(ct2)


@pytest.mark.parametrize(
    "mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps",
    ([5, 0.8, 3.87e-3, [[1400, 17]], 4, 2, 50],),
)
def test_evolve(
    mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps
):
    ph_list = [
        Phonon.simple_phonon(
            Quantity(omega, "cm^{-1}"), Quantity(displacement, "a.u."), ph_phys_dim
        )
        for omega, displacement in ph_info
    ]
    mol_list = MolList(
        [Mol(Quantity(elocalex_value, "a.u."), ph_list)] * mol_num,
        Quantity(j_constant_value, "eV"),
    )
    ct1 = ChargeTransport(mol_list)
    half_nsteps = nsteps // 2
    ct1.evolve(evolve_dt, half_nsteps)
    ct1.evolve(evolve_dt, nsteps - half_nsteps)
    ct2 = ChargeTransport(mol_list)
    ct2.evolve(evolve_dt, nsteps)
    assert ct1.is_similar(ct2)
    assert ct1.get_dump_dict() == ct2.get_dump_dict()


@pytest.mark.parametrize(
    "mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps",
    ([3, 1, 3.87e-3, [[1e-5, 1e-5]], 2, 2, 50],),
)
def test_band_limit_finite_t(
    mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps
):
    ph_list = [
        Phonon.simple_phonon(
            Quantity(omega, "cm^{-1}"), Quantity(displacement, "a.u."), ph_phys_dim
        )
        for omega, displacement in ph_info
    ]
    mol_list = MolList(
        [Mol(Quantity(elocalex_value, "a.u."), ph_list)] * mol_num,
        Quantity(j_constant_value, "eV"),
    )
    ct1 = ChargeTransport(mol_list)
    ct1.evolve(evolve_dt, nsteps)
    ct2 = ChargeTransport(mol_list, temperature=low_t)
    ct2.evolve(evolve_dt, nsteps)
    assert ct1.is_similar(ct2)
