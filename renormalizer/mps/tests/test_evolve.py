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
    "method, evolve_dt, use_rk, rtol",
    (
        # [EvolveMethod.tdvp_mctdh, 2.0, 1e-2],
        [EvolveMethod.tdvp_mctdh_new, 2.0, None, 1e-2],
        [EvolveMethod.tdvp_ps, 15.0, True, 1e-2],
        [EvolveMethod.tdvp_ps, 15.0, False, 1e-2],
    ),
)
def test_ZeroTcorr_TDVP(method, evolve_dt, use_rk, rtol):
    # procedure = [[50, 0], [50, 0], [50, 0]]
    procedure = [[20, 0], [20, 0], [20, 0]]
    optimize_config = OptimizeConfig(procedure=procedure)

    mol_list = parameter.mol_list

    evolve_config = EvolveConfig(method, evolve_dt=evolve_dt, adaptive=False)
    evolve_config.tdvp_ps_rk4 = use_rk

    zero_t_corr = SpectraTwoWayPropZeroT(
        mol_list,
        "abs",
        optimize_config,
        evolve_config=evolve_config,
        offset=Quantity(2.28614053, "ev"),
    )
    zero_t_corr.info_interval = 30
    nsteps = 200
    # nsteps = 1200
    zero_t_corr.evolve(evolve_dt, nsteps)
    with open(
        os.path.join(
            cur_dir, "zero_t_%s.npy" % str(evolve_config.method).split(".")[1]
        ),
        "rb",
    ) as f:
        ZeroTabs_std = np.load(f)
    assert np.allclose(zero_t_corr.autocorr[:nsteps], ZeroTabs_std[:nsteps], rtol=rtol)

# from matplotlib import pyplot as plt
#
# plt.clf()
# plt.plot(zero_t_corr.autocorr[:nsteps:2], label="c")
# plt.plot(ZeroTabs_std[:nsteps:2], label="tdvp svd")
# with open("/home/wtli/GitClone/renormalizer/renormalizer/spectra/tests/ZeroTabs_2svd.npy", "rb") as fin:
#     data = np.load(fin)
# plt.plot(data, label="pc std")
# plt.legend()
# plt.savefig("a.png")

@pytest.mark.parametrize(
    "method, nsteps, evolve_dt, use_rk, rtol, interval",
    (
       [EvolveMethod.tdvp_mctdh_new, 85, 4, None, 1e-2, 2],
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
