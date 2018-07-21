# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
from __future__ import absolute_import, print_function, division

import unittest
import os

import numpy as np
from ddt import ddt, data

from ephMPS.spectra import spectra
from ephMPS.tests import parameter
from ephMPS import constant
from ephMPS.mps import solver
from ephMPS.mps.mpo import Mpo, Mps
from ephMPS.spectra.tests import cur_dir


@ddt
class TestZeroExactEmi(unittest.TestCase):
    @data([[[4, 4]], 1e-3])
    def test_zero_exact_emi(self, value):
        # print "data", value
        nexciton = 1
        procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]

        mps, mpo = solver.construct_mps_mpo_2(parameter.mol_list, parameter.j_matrix, procedure[0][0], nexciton)
        solver.optimize_mps(mps, mpo, procedure, method="2site")

        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in range(mpo.pbond_list[0]):
            mpo[0][0, ibra, ibra, 0] -= 2.28614053 / constant.au2ev

        dipole_mpo = Mpo.onsite(parameter.mol_list, mpo.pbond_list, "a", dipole=True)
        nsteps = 3000
        dt = 30.0
        temperature = 0
        autocorr = spectra.exact_spectra("emi", parameter.mol_list, mps, dipole_mpo, nsteps, dt, temperature)
        autocorr = np.array(autocorr)
        with open(os.path.join(cur_dir, 'ZeroExactEmi.npy'), 'rb') as fin:
            std = np.load(fin)

        self.assertTrue(np.allclose(autocorr, std, rtol=value[1]))


@ddt
class TestZeroTCorr(unittest.TestCase):
    # todo: variational method not implemented yet

    @data(
        [1, "svd", True, [[4, 4]], 1e-3],
        [2, "svd", True, [[4, 4]], 1e-3],
        # [1, "svd", None, [[4, 4]], 1e-3],
        # [2, "svd", None, [[4, 4]], 1e-3],
        # [1, "variational", None, [[4, 4]], 1e-2],
        # [2, "variational", None, [[4, 4]], 1e-2],
        [1, "svd", True, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
        [2, "svd", True, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
        # [1, "svd", None, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
        # [2, "svd", None, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
        # [1, "variational", None, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
        # [2, "variational", None, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2]
    )
    def test_zero_t_corr(self, value):
        np.random.seed(0)
        # print "data", value
        nexciton = 0
        procedure = [[1, 0], [1, 0], [1, 0]]
        nsteps = 100
        dt = 30.0
        mol_list = parameter.custom_mol_list(*value[3])
        autocorr = spectra.calc_zero_t_corr(mol_list, parameter.j_matrix, procedure, nexciton, 2, 2.28614053, nsteps,
                                            dt,
                                            value[0], value[1])
        with open(os.path.join(cur_dir, 'ZeroTabs_' + str(value[0]) + str(value[1]) + '.npy'), 'rb') as f:
            std = np.load(f)
        self.assertTrue(np.allclose(autocorr, std, rtol=value[4]))

    @data([1, "svd", True, [[4, 4]], 1e-3],
          [2, "svd", True, [[4, 4]], 1e-3],
          # [1, "svd", None, [[4, 4]], 1e-3],
          # [2, "svd", None, [[4, 4]], 1e-3],
          # [1, "variational", None, [[4, 4]], 1e-3],
          # [2, "variational", None, [[4, 4]], 1e-3],
          [1, "svd", True, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
          [2, "svd", True, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
          # [1, "svd", None, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
          # [2, "svd", None, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
          # [1, "variational", None, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
          # [2, "variational", None, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2]
          )
    def test_zero_t_corr_mposcheme3(self, value):
        np.random.seed(0)
        # print "data", value
        j_matrix = np.array([[0.0, -0.1, 0.0], [-0.1, 0.0, -0.3], [0.0, -0.3, 0.0]]) / constant.au2ev
        nexciton = 0
        procedure = [[1, 0], [1, 0], [1, 0]]
        mol_list = parameter.custom_mol_list(*value[3])
        nsteps = 50
        dt = 30.0
        autocorr2 = spectra.calc_zero_t_corr(mol_list, j_matrix, procedure, nexciton, 2, 2.28614053, nsteps, dt,
                                             value[0], value[1])
        autocorr3 = spectra.calc_zero_t_corr(mol_list, j_matrix, procedure, nexciton, 3, 2.28614053, nsteps, dt,
                                             value[0], value[1])
        self.assertTrue(np.allclose(autocorr2, autocorr3, rtol=value[4]))


@ddt
class TestFiniteTSpectraEmi(unittest.TestCase):
    @data([2, "svd", True, [[4, 4]], 1e-3],
          [2, "svd", True, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2]
          )
    def test_FiniteT_spectra_emi(self, value):
        # print "data", value
        nexciton = 1
        procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]
        mol_list = parameter.custom_mol_list(*value[3])
        i_mps, h_mpo = solver.construct_mps_mpo_2(mol_list, parameter.j_matrix, procedure[0][0], nexciton)

        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in range(h_mpo.pbond_list[0]):
            h_mpo[0][0, ibra, ibra, 0] -= 2.28614053 / constant.au2ev

        dipole_mpo = Mpo.onsite(mol_list, h_mpo.pbond_list, 'a', dipole=True)
        nsteps = 30
        dt = 30.0
        ex_mpo = Mpo.max_entangled_ex(mol_list, h_mpo.pbond_list)

        insteps = 50
        autocorr = spectra.finite_t_spectra("emi", mol_list, ex_mpo, h_mpo, dipole_mpo, nsteps, dt, insteps,
                                            temperature=298, algorithm=value[0], compress_method=value[1])

        with open(os.path.join(cur_dir, 'TTemi_' + str(value[0]) + str(value[1]) + ".npy"), 'rb') as fin:
            std = np.load(fin)
        self.assertTrue(np.allclose(autocorr, std[0:nsteps], rtol=value[4]))


@ddt
class TestFiniteTSpectraAbs(unittest.TestCase):
    @data(
        [2, "svd", True, [[4, 4]], 1e-3],
        [2, "svd", True, [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2]
    )
    def test_finite_t_spectra_abs(self, value):
        # print "data", value
        nexciton = 0
        procedure = [[1, 0], [1, 0], [1, 0]]
        mol_list = parameter.custom_mol_list(*value[3])
        i_mps, h_mpo = solver.construct_mps_mpo_2(mol_list, parameter.j_matrix, procedure[0][0], nexciton)

        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in range(h_mpo.pbond_list[0]):
            h_mpo[0][0, ibra, ibra, 0] -= 2.28614053 / constant.au2ev

        dipole_mpo = Mpo.onsite(mol_list, h_mpo.pbond_list, "a^\dagger", dipole=True)
        gs_mps = Mps.max_entangled_gs(mol_list, h_mpo.pbond_list)
        gs_mpo = Mpo.from_mps(gs_mps)

        nsteps = 50
        dt = 30.0
        autocorr = spectra.finite_t_spectra("abs", mol_list, gs_mpo, h_mpo, dipole_mpo, nsteps, dt,
                                            temperature=298, algorithm=value[0], compress_method=value[1])

        with open(os.path.join(cur_dir, "TTabs_" + str(value[1]) + ".npy"), 'rb') as fin:
            std = np.load(fin)

        self.assertTrue(np.allclose(autocorr, std[0:nsteps], rtol=value[4]))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestZeroExactEmi)
    unittest.TextTestRunner().run(suite)
    #unittest.main()
