# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
from __future__ import absolute_import, print_function, division

import os

import numpy as np
import pytest

from renormalizer.spectra import (
    SpectraExact,
    SpectraOneWayPropZeroT,
    SpectraTwoWayPropZeroT,
    SpectraFiniteT,
)
from renormalizer.spectra.tests import cur_dir
from renormalizer.tests import parameter
from renormalizer.utils import constant, Quantity, OptimizeConfig


def test_zero_exact_emi():
    # print "data", value
    mol_list = parameter.mol_list

    exact_emi = SpectraExact(mol_list, "emi")
    # setup a large interval because evaluating expectations are expensive when evolution is fast
    exact_emi.info_interval = 100
    nsteps = 3000
    # nsteps = 100
    dt = 30.0
    exact_emi.evolve(dt, nsteps)

    with open(os.path.join(cur_dir, "ZeroExactEmi.npy"), "rb") as fin:
        std = np.load(fin)

    assert np.allclose(exact_emi.autocorr[:nsteps], std[:nsteps], rtol=1e-3)


@pytest.mark.parametrize(
    "algorithm, compress_method, ph_info, rtol",
    (
        [1, "svd", [[4, 4]], 1e-2],
        [2, "svd", [[4, 4]], 1e-2],
        [1, "svd", [[4, 4], [2, 2], [1.0e-7, 1.0e-7]], 1e-2],
        [2, "svd", [[4, 4], [2, 2], [1.0e-7, 1.0e-7]], 1e-2],
    ),
)
def test_zero_t_abs(algorithm, compress_method, ph_info, rtol, switch_to_64backend):
    np.random.seed(0)
    # print "data", value
    procedure = [[1, 0], [1, 0], [1, 0]]
    optimize_config = OptimizeConfig()
    optimize_config.procedure = procedure
    mol_list = parameter.custom_mol_list(None, *ph_info)
    if algorithm == 1:
        SpectraZeroT = SpectraOneWayPropZeroT
    else:
        SpectraZeroT = SpectraTwoWayPropZeroT

    zero_t_corr = SpectraZeroT(
        mol_list.switch_scheme(2), "abs", optimize_config, offset=parameter.offset
    )
    zero_t_corr.info_interval = 30
    nsteps = 100
    dt = 30.0
    zero_t_corr.evolve(dt, nsteps)
    with open(
        os.path.join(
            cur_dir, "ZeroTabs_" + str(algorithm) + str(compress_method) + ".npy"
        ),
        "rb",
    ) as f:
        std = np.load(f)
    assert np.allclose(zero_t_corr.autocorr[:nsteps], std[:nsteps], rtol=rtol)


@pytest.mark.parametrize(
    "algorithm, compress_method, ph_info, rtol",
    (
        [1, "svd", [[4, 4]], 1e-3],
        [2, "svd", [[4, 4]], 1e-3],
        [1, "svd", [[4, 4], [2, 2], [1.0e-7, 1.0e-7]], 1e-2],
        [2, "svd", [[4, 4], [2, 2], [1.0e-7, 1.0e-7]], 1e-2],
    ),
)
def test_zero_t_abs_mposcheme3(
    algorithm, compress_method, ph_info, rtol, switch_to_64backend
):
    np.random.seed(0)
    # print "data", value
    j_matrix = (
        np.array([[0.0, -0.1, 0.0], [-0.1, 0.0, -0.3], [0.0, -0.3, 0.0]])
        / constant.au2ev
    )
    procedure = [[1, 0], [1, 0], [1, 0]]
    mol_list = parameter.custom_mol_list(j_matrix, *ph_info)
    nsteps = 50
    dt = 30.0
    if algorithm == 1:
        SpectraZeroT = SpectraOneWayPropZeroT
    else:
        SpectraZeroT = SpectraTwoWayPropZeroT
    optimize_config = OptimizeConfig()
    optimize_config.procedure = procedure
    zero_t_corr2 = SpectraZeroT(
        mol_list, "abs", optimize_config, offset=parameter.offset
    )
    zero_t_corr2.evolve(dt, nsteps)
    zero_t_corr3 = SpectraZeroT(
        mol_list.switch_scheme(3), "abs", optimize_config, offset=parameter.offset
    )
    zero_t_corr3.evolve(dt, nsteps)

    assert np.allclose(zero_t_corr2.autocorr, zero_t_corr3.autocorr, rtol=rtol)


@pytest.mark.parametrize("algorithm, rtol", ([1, 1e-2], [2, 1e-2]))
def test_zero_t_emi(algorithm, rtol):
    np.random.seed(0)
    # print "data", value
    mol_list = parameter.mol_list
    if algorithm == 1:
        SpectraZeroT = SpectraOneWayPropZeroT
    else:
        SpectraZeroT = SpectraTwoWayPropZeroT

    zero_t_corr = SpectraZeroT(mol_list, "emi")
    zero_t_corr.info_interval = 50
    nsteps = 100
    dt = 30.0
    zero_t_corr.evolve(dt, nsteps)
    with open(os.path.join(cur_dir, "ZeroExactEmi.npy"), "rb") as f:
        std = np.load(f)
    assert np.allclose(zero_t_corr.autocorr[:nsteps], std[:nsteps], rtol=rtol)


@pytest.mark.parametrize(
    "algorithm, compress_method, ph_info, rtol",
    ([2, "svd", [[4, 4]], 1e-2], [2, "svd", [[4, 4], [2, 2], [1.0e-7, 1.0e-7]], 1e-2]),
)
def test_finite_t_spectra_emi(algorithm, compress_method, ph_info, rtol):
    np.random.seed(0)
    # print "data", value
    mol_list = parameter.custom_mol_list(None, *ph_info)
    insteps = 50
    finite_t_emi = SpectraFiniteT(
        mol_list, "emi", Quantity(298, "K"), insteps, parameter.offset
    )
    nsteps = 30
    dt = 30.0
    finite_t_emi.evolve(dt, nsteps)
    with open(
        os.path.join(
            cur_dir, "TTemi_" + str(algorithm) + str(compress_method) + ".npy"
        ),
        "rb",
    ) as fin:
        std = np.load(fin)
    assert np.allclose(finite_t_emi.autocorr[:nsteps], std[:nsteps], rtol=rtol)


@pytest.mark.parametrize(
    "algorithm, compress_method, ph_info, rtol",
    ([2, "svd", [[4, 4]], 1e-2], [2, "svd", [[4, 4], [2, 2], [1.0e-7, 1.0e-7]], 1e-2]),
)
def test_finite_t_spectra_abs(algorithm, compress_method, ph_info, rtol):
    # print "data", value
    mol_list = parameter.custom_mol_list(None, *ph_info)
    insteps = 50
    finite_t_abs = SpectraFiniteT(
        mol_list, "abs", Quantity(298, "K"), insteps, parameter.offset
    )
    nsteps = 50
    dt = 30.0
    finite_t_abs.evolve(dt, nsteps)
    with open(
        os.path.join(cur_dir, "TTabs_" + str(compress_method) + ".npy"), "rb"
    ) as fin:
        std = np.load(fin)
    assert np.allclose(finite_t_abs.autocorr[:nsteps], std[:nsteps], rtol=rtol)
