# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import unittest
import os

from ddt import ddt, data, unpack
import numpy as np

from ephMPS.spectra import SpectraTwoWayPropZeroT, SpectraFiniteT
from ephMPS.utils import OptimizeConfig, EvolveMethod, EvolveConfig, Quantity
from ephMPS.tests import parameter
from ephMPS.mps.tests import cur_dir


@ddt
class TestZeroT(unittest.TestCase):

    @data(
        #[EvolveMethod.tdvp_mctdh, 2.0, 1e-2], # not working
        [EvolveMethod.tdvp_mctdh_new, 2.0, 1e-2],
        [EvolveMethod.tdvp_ps, 15.0, 1e-2],
    )
    @unpack
    def test_ZeroTcorr_TDVP(self, method, evolve_dt, rtol):

        # procedure = [[50, 0], [50, 0], [50, 0]]
        procedure = [[20, 0], [20, 0], [20, 0]]
        optimize_config = OptimizeConfig()
        optimize_config.procedure = procedure

        mol_list = parameter.mol_list

        evolve_config = EvolveConfig()
        evolve_config.scheme = method

        zero_t_corr = SpectraTwoWayPropZeroT(mol_list, "abs", optimize_config, evolve_config=evolve_config, offset=Quantity(2.28614053, 'ev'))
        zero_t_corr.info_interval = 30
        nsteps = 200
        zero_t_corr.evolve(evolve_dt, nsteps)
        with open(os.path.join(cur_dir, "zero_t_%s.npy" % str(evolve_config.scheme).split('.')[1]), 'rb') as f:
            ZeroTabs_std = np.load(f)
        self.assertTrue(np.allclose(zero_t_corr.autocorr[:nsteps], ZeroTabs_std[:nsteps], rtol=rtol))
        # from matplotlib import pyplot as plt
        # plt.plot(zero_t_corr.autocorr)
        # plt.plot(ZeroTabs_std)
        # plt.show()

@ddt
class TestFiniteT(unittest.TestCase):
    @data(
        [EvolveMethod.tdvp_mctdh_new, 191, 2.0, 1e-2],
        [EvolveMethod.tdvp_ps, 30, 30, 1e-2]
    )
    @unpack
    def test_FiniteT_spectra_emi(self, method, nsteps, evolve_dt, rtol):
        mol_list = parameter.mol_list
        temperature = Quantity(298, 'K')
        offset = Quantity(2.28614053, 'ev')
        evolve_config = EvolveConfig()
        evolve_config.scheme = method
        finite_t_corr = SpectraFiniteT(mol_list, "emi", temperature, 50, offset, evolve_config=evolve_config)
        finite_t_corr.evolve(evolve_dt, nsteps)
        with open(os.path.join(cur_dir, "finite_t_%s.npy" % str(evolve_config.scheme).split('.')[1]), 'rb') as fin:
            std = np.load(fin)
        self.assertTrue(np.allclose(finite_t_corr.autocorr[:nsteps], std[:nsteps], rtol=rtol))
        # from matplotlib import pyplot as plt
        # plt.plot(finite_t_corr.autocorr)
        # plt.plot(std)
        # plt.show()

if __name__ == "__main__":
    print("Test RK")
    unittest.main()
