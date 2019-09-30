# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import os

import pytest
import numpy as np

from renormalizer.spectra import SpectraTwoWayPropZeroT, SpectraFiniteT
from renormalizer.utils import OptimizeConfig, EvolveMethod, EvolveConfig, Quantity
from renormalizer.tests import parameter
from renormalizer.mps.tests import cur_dir


@pytest.mark.parametrize(
    "method, evolve_dt, nsteps, use_rk, cmf_or_midpoint, rtol, interval",
    (
        [EvolveMethod.tdvp_vmf, 15., 100, False, None, 1e-2, 1],
        [EvolveMethod.tdvp_mu_switch_gauge, 8, 50, None, False, 1e-2, 4],
        [EvolveMethod.tdvp_mu_switch_gauge, 4, 100, None, True, 1e-2, 2],
        [EvolveMethod.tdvp_mu_fixed_gauge, 6, 70, None, False, 1e-2, 3],
        [EvolveMethod.tdvp_mu_fixed_gauge, 12, 35, None, True, 1e-2, 6],
        [EvolveMethod.tdvp_ps, 15.0, 200, True, None, 1e-2, 1],
        [EvolveMethod.tdvp_ps, 15.0, 200, False, None, 1e-2, 1],
        [EvolveMethod.tdvp_mu_vmf, 15.0, 100, False, None, 1e-2, 1],
    ),
)
def test_ZeroTcorr_TDVP(method, evolve_dt, nsteps, use_rk, cmf_or_midpoint, rtol, interval):
    procedure = [[20, 0], [20, 0], [20, 0]]
    optimize_config = OptimizeConfig(procedure=procedure)

    mol_list = parameter.mol_list

    evolve_config = EvolveConfig(method, evolve_dt=evolve_dt, adaptive=False)
    evolve_config.tdvp_ps_rk4 = use_rk
    evolve_config.tdvp_mctdh_cmf = cmf_or_midpoint
    evolve_config.tdvp_mu_midpoint = cmf_or_midpoint
    if method is EvolveMethod.tdvp_vmf:
        evolve_config.reg_epsilon = 1e-5

    zero_t_corr = SpectraTwoWayPropZeroT(
        mol_list,
        "abs",
        optimize_config,
        evolve_config=evolve_config,
        offset=Quantity(2.28614053, "ev"),
    )
    zero_t_corr.info_interval = 30
    zero_t_corr.evolve(evolve_dt, nsteps)
    file_name_mapping = {
        EvolveMethod.tdvp_mu_switch_gauge: "zero_t_tdvp_mu.npy",
        EvolveMethod.tdvp_mu_fixed_gauge: "zero_t_tdvp_mu.npy",
        EvolveMethod.tdvp_mu_vmf: "zero_t_tdvp_ps.npy",
        EvolveMethod.tdvp_vmf: "zero_t_tdvp_ps.npy",
        EvolveMethod.tdvp_ps: "zero_t_tdvp_ps.npy"
    }
    fname = file_name_mapping[method]
    with open(os.path.join(cur_dir, fname),"rb") as f:
        std = np.load(f)
    assert np.allclose(zero_t_corr.autocorr[:nsteps], std[:interval*nsteps:interval], rtol=rtol)


@pytest.mark.parametrize(
    "method, nsteps, evolve_dt, use_rk, rtol, interval",
    (
        [EvolveMethod.tdvp_vmf, 30, 6.,False, 1e-2, 3],
        [EvolveMethod.tdvp_mu_switch_gauge, 10, 32, None, 1e-2, 16],
        [EvolveMethod.tdvp_mu_fixed_gauge, 5, 64, None, 1e-2, 32],
        [EvolveMethod.tdvp_ps, 30, 30, True, 1e-2, 1],
        [EvolveMethod.tdvp_ps, 30, 30, False, 1e-2, 1],
        [EvolveMethod.tdvp_mu_vmf, 30, 6, False, 1e-2, 3],
    ),
)
def test_finite_t_spectra_emi_TDVP(method, nsteps, evolve_dt, use_rk, rtol, interval):
    mol_list = parameter.mol_list
    temperature = Quantity(298, "K")
    offset = Quantity(2.28614053, "ev")
    evolve_config = EvolveConfig(method)
    evolve_config.tdvp_ps_rk4 = use_rk
    if method is EvolveMethod.tdvp_vmf:
        evolve_config.reg_epsilon = 1e-5
    
    finite_t_corr = SpectraFiniteT(
        mol_list, "emi", temperature, 50, offset, evolve_config=evolve_config
    )
    finite_t_corr.evolve(evolve_dt, nsteps)
    
    file_name_mapping = {
        EvolveMethod.tdvp_mu_switch_gauge: "finite_t_tdvp_mu.npy",
        EvolveMethod.tdvp_mu_fixed_gauge: "finite_t_tdvp_mu.npy",
        EvolveMethod.tdvp_ps: "finite_t_tdvp_ps.npy",
        EvolveMethod.tdvp_mu_vmf: "finite_t_tdvp_mu.npy",
        EvolveMethod.tdvp_vmf: "finite_t_tdvp_mu.npy"
    }
    fname = file_name_mapping[method]
    with open(os.path.join(cur_dir, fname),"rb") as f:
        std = np.load(f)
    assert np.allclose(
        finite_t_corr.autocorr[:nsteps], std[:interval*nsteps:interval], rtol=rtol
    )
    # from matplotlib import pyplot as plt
    # plt.plot(finite_t_corr.autocorr)
    # plt.plot(std)
    # plt.show()
