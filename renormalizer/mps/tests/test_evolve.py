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
    "method, evolve_dt, nsteps, use_rk, cmf, rtol, interval",
    (
        # [EvolveMethod.tdvp_mctdh, 2.0, 1e-2],
        [EvolveMethod.tdvp_mctdh_new, 6, 70, None, False, 1e-2, 3],
        [EvolveMethod.tdvp_mctdh_new, 2, 200, None, True, 1e-2, 1],
        [EvolveMethod.tdvp_ps, 15.0, 200, True, None, 1e-2, 1],
        [EvolveMethod.tdvp_ps, 15.0, 200, False, None, 1e-2, 1],
    ),
)
def test_ZeroTcorr_TDVP(method, evolve_dt, nsteps, use_rk, cmf, rtol, interval):
    procedure = [[20, 0], [20, 0], [20, 0]]
    optimize_config = OptimizeConfig(procedure=procedure)

    mol_list = parameter.mol_list

    evolve_config = EvolveConfig(method, evolve_dt=evolve_dt, adaptive=False)
    evolve_config.tdvp_ps_rk4 = use_rk
    evolve_config.tdvp_mu_cmf = cmf

    zero_t_corr = SpectraTwoWayPropZeroT(
        mol_list,
        "abs",
        optimize_config,
        evolve_config=evolve_config,
        offset=Quantity(2.28614053, "ev"),
    )
    zero_t_corr.info_interval = 30
    zero_t_corr.evolve(evolve_dt, nsteps)
    with open(
        os.path.join(
            cur_dir, "zero_t_%s.npy" % str(evolve_config.method).split(".")[1]
        ),
        "rb",
    ) as f:
        std = np.load(f)
    assert np.allclose(zero_t_corr.autocorr[:nsteps], std[:interval*nsteps:interval], rtol=rtol)


@pytest.mark.parametrize(
    "method, nsteps, evolve_dt, use_rk, rtol, interval",
    (
       [EvolveMethod.tdvp_mctdh_new, 10, 32, None, 1e-2, 16],
       [EvolveMethod.tdvp_ps, 30, 30, True, 1e-2, 1],
       [EvolveMethod.tdvp_ps, 30, 30, False, 1e-2, 1],
    ),
)
def test_finite_t_spectra_emi_TDVP(method, nsteps, evolve_dt, use_rk, rtol, interval):
    mol_list = parameter.mol_list
    temperature = Quantity(298, "K")
    offset = Quantity(2.28614053, "ev")
    evolve_config = EvolveConfig(method)
    evolve_config.tdvp_ps_rk4 = use_rk
    finite_t_corr = SpectraFiniteT(
        mol_list, "emi", temperature, 50, offset, evolve_config=evolve_config
    )
    finite_t_corr.evolve(evolve_dt, nsteps)
    with open(
        os.path.join(
            cur_dir, "finite_t_%s.npy" % str(evolve_config.method).split(".")[1]
        ),
        "rb",
    ) as fin:
        std = np.load(fin)
    assert np.allclose(
        finite_t_corr.autocorr[:nsteps], std[:interval*nsteps:interval], rtol=rtol
    )
    # from matplotlib import pyplot as plt
    # plt.plot(finite_t_corr.autocorr)
    # plt.plot(std)
    # plt.show()
