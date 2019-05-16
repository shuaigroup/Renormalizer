# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

import os

import numpy as np
import pytest

from ephMPS.model import Phonon, Mol, MolList
from ephMPS.mps import Mps, Mpo
from ephMPS.mps.solver import optimize_mps
from ephMPS.transport import ChargeTransport
from ephMPS.utils import Quantity
from ephMPS.utils import (
    BondDimDistri,
    CompressCriteria,
    CompressConfig,
    EvolveMethod,
    EvolveConfig,
)
from ephMPS.transport.tests.band_param import band_limit_mol_list, assert_band_limit, low_t


def test_init_state():
    ph = Phonon.simple_phonon(Quantity(1), Quantity(1), 10)
    mol_list = MolList([Mol(Quantity(0), [ph])], Quantity(0), scheme=3)
    mpo = Mpo(mol_list)
    mps = Mps.random(mol_list, 1, 10)
    optimize_mps(mps, mpo)
    ct = ChargeTransport(mol_list)
    assert mps.angle(ct.latest_mps) == pytest.approx(1)


@pytest.mark.parametrize(
    "method, evolve_dt, nsteps, rtol",
    (
        (EvolveMethod.prop_and_compress, 4, 25, 1e-3),
        # not working. Moves slightly slower. Dunno why.
        # (EvolveMethod.tdvp_mctdh_new, 2, 50, 1e-2),
        (EvolveMethod.tdvp_ps, 2, 100, 1e-3),
    ),
)
@pytest.mark.parametrize(
    "scheme", (3, 4)
)
def test_bandlimit_zero_t(method, evolve_dt, nsteps, rtol, scheme):
    np.random.seed(0)
    evolve_config = EvolveConfig(method)
    ct = ChargeTransport(band_limit_mol_list.switch_scheme(scheme), evolve_config=evolve_config)
    ct.stop_at_edge = True
    ct.evolve(evolve_dt, nsteps)
    assert_band_limit(ct, rtol)


@pytest.mark.parametrize(
    "method", (
            EvolveMethod.prop_and_compress,
            EvolveMethod.tdvp_ps,
    )
)
@pytest.mark.parametrize("init_dt", (1e-1, 20))
def test_adaptive_zero_t(method, init_dt):
    np.random.seed(0)
    evolve_config = EvolveConfig(method=method, evolve_dt=init_dt, adaptive=True)
    ct = ChargeTransport(
        band_limit_mol_list, evolve_config=evolve_config, stop_at_edge=True
    )
    ct.evolve()
    assert_band_limit(ct, 1e-2)


def test_32backend(switch_to_32backend):
    evolve_config = EvolveConfig(evolve_dt=4, adaptive=True)
    ct = ChargeTransport(band_limit_mol_list, evolve_config=evolve_config)
    ct.stop_at_edge = True
    ct.evolve()
    assert_band_limit(ct, 1e-2)


def test_gaussian_bond_dim():
    compress_config = CompressConfig(
        criteria=CompressCriteria.fixed,
        bonddim_distri=BondDimDistri.center_gauss,
        max_bonddim=10,
    )
    evolve_config = EvolveConfig(evolve_dt=4, adaptive=True)
    ct = ChargeTransport(
        band_limit_mol_list,
        compress_config=compress_config,
        evolve_config=evolve_config,
    )
    ct.stop_at_edge = True
    ct.evolve()
    assert_band_limit(ct, 1e-2)


def assert_iterable_equal(i1, i2):
    if isinstance(i1, str):
        assert i1 == i2
        return
    if not hasattr(i1, "__iter__"):
        if isinstance(i1, float):
            assert i1 == pytest.approx(i2)
        else:
            assert i1 == i2
        return
    for ii1, ii2 in zip(i1, i2):
        assert_iterable_equal(ii1, ii2)


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
        scheme=3
    )
    ct1 = ChargeTransport(mol_list)
    ct1.evolve(evolve_dt, nsteps)
    ct2 = ChargeTransport(mol_list)
    ct2.economic_mode = True
    ct2.evolve(evolve_dt, nsteps)
    assert ct1.is_similar(ct2)

    assert_iterable_equal(ct1.get_dump_dict(), ct2.get_dump_dict())


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
        scheme=3
    )
    compress_config = CompressConfig(
        threshold=1e-5
    )  # make the size of the MPS grow fast
    evolve_config = EvolveConfig(memory_limit="100 KB")
    ct1 = ChargeTransport(
        mol_list,
        evolve_config=evolve_config,
        compress_config=compress_config,
        stop_at_edge=False,
    )
    ct1.evolve(evolve_dt, nsteps)
    ct2 = ChargeTransport(mol_list, compress_config=compress_config, stop_at_edge=False)
    ct2.evolve(evolve_dt, nsteps)
    assert ct1.is_similar(ct2, rtol=1e-2)
    assert ct1.latest_mps.peak_bytes < ct2.latest_mps.peak_bytes


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
        scheme=3
    )
    ct1 = ChargeTransport(mol_list, temperature=Quantity(298, "K"))
    ct1.reduced_density_matrices = None
    ct1.evolve(evolve_dt, nsteps)
    ct2 = ChargeTransport(mol_list, temperature=Quantity(298, "K"))
    ct2.reduced_density_matrices = None
    ct2.latest_mps.compress_add = True
    ct2.evolve(evolve_dt, nsteps)
    assert ct1.is_similar(ct2, rtol=1e-2)


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
        scheme=3
    )
    ct = ChargeTransport(mol_list, temperature=Quantity(temperature, "K"), stop_at_edge=False)
    ct.evolve(evolve_dt, nsteps)
    for mps in ct.tdmps_list:
        rdm = mps.calc_reduced_density_matrix()
        # best we can do?
        assert np.allclose(np.diag(rdm), mps.e_occupations)


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
        scheme=3
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
        scheme=3
    )
    ct1 = ChargeTransport(mol_list, stop_at_edge=False)
    half_nsteps = nsteps // 2
    ct1.evolve(evolve_dt, half_nsteps)
    ct1.evolve(evolve_dt, nsteps - half_nsteps)
    ct2 = ChargeTransport(mol_list, stop_at_edge=False)
    ct2.evolve(evolve_dt, nsteps)
    assert ct1.is_similar(ct2)
    assert_iterable_equal(ct1.get_dump_dict(), ct2.get_dump_dict())

    # test dump
    ct2.dump_dir = '.'
    ct2.job_name = 'test'
    ct2.dump_dict()
    os.remove('test.json')


@pytest.mark.parametrize(
    "mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps",
    ([3, 1, 3.87e-3, [[1e-5, 1e-5]], 2, 2, 50],),
)
@pytest.mark.parametrize(
    "scheme", (3, 4)
)
def test_band_limit_finite_t(
    mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps, scheme
):
    ph_list = [
        Phonon.simple_phonon(
            Quantity(omega, "cm^{-1}"), Quantity(displacement, "a.u."), ph_phys_dim
        )
        for omega, displacement in ph_info
    ]
    mol_list = MolList(
        [Mol(Quantity(elocalex_value, "a.u."), ph_list)] * mol_num,
        Quantity(j_constant_value, "eV"), scheme=scheme
    )
    ct1 = ChargeTransport(mol_list, stop_at_edge=False)
    ct1.evolve(evolve_dt, nsteps)
    ct2 = ChargeTransport(mol_list, temperature=low_t, stop_at_edge=False)
    ct2.evolve(evolve_dt, nsteps)
    assert ct1.is_similar(ct2)


@pytest.mark.parametrize(
    "mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps",
    ([3, 1, 3.87e-3, [[1400, 17]], 8, 2, 50],),
)
def test_scheme4_finite_t(
    mol_num, j_constant_value, elocalex_value, ph_info, ph_phys_dim, evolve_dt, nsteps
):
    temperature = Quantity(1, "a.u.")
    ph_list = [
        Phonon.simple_phonon(
            Quantity(omega, "cm^{-1}"), Quantity(displacement, "a.u."), ph_phys_dim
        )
        for omega, displacement in ph_info
    ]
    mol_list = MolList(
        [Mol(Quantity(elocalex_value, "a.u."), ph_list)] * mol_num,
        Quantity(j_constant_value, "eV")
    )
    ct1 = ChargeTransport(mol_list.switch_scheme(3), temperature=temperature, stop_at_edge=False)
    ct1.evolve(evolve_dt, nsteps)
    ct2 = ChargeTransport(mol_list.switch_scheme(4), temperature=temperature, stop_at_edge=False)
    ct2.evolve(evolve_dt, nsteps)
    assert ct1.is_similar(ct2, rtol=1e-2)
