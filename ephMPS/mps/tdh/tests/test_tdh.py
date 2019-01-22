# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import logging
import unittest

import numpy as np
from ddt import ddt

from ephMPS.model import MolList, Mol
from ephMPS.mps import Mpo
from ephMPS.mps.solver import construct_mps_mpo_2, optimize_mps
from ephMPS.mps.tdh import tdh
from ephMPS.tests.parameter import hartree_mol_list, ph_phys_dim, custom_mol_list, \
    hartree_ph_list, elocalex, dipole_abs
from ephMPS.utils import Quantity, log, constant


@ddt
class Test_TDH(unittest.TestCase):

    def test_SCF(self):

        #  EX
        nexciton = 1
        WFN, Etot = tdh.SCF(hartree_mol_list, nexciton)
        self.assertAlmostEqual(Etot, 0.0843103276663)

        fe, fv = 1, 6
        HAM, Etot, A_el = tdh.construct_H_Ham(hartree_mol_list, nexciton, WFN, fe, fv, debug=True)
        self.assertAlmostEqual(Etot, 0.0843103276663)
        occ_std = np.array([[0.20196397], [0.35322702], [0.444809]])
        self.assertTrue(np.allclose(A_el, occ_std))

        # GS
        nexciton = 0
        WFN, Etot = tdh.SCF(hartree_mol_list, nexciton)
        self.assertAlmostEqual(Etot, 0.0)

        fe, fv = 1, 6
        HAM, Etot, A_el = tdh.construct_H_Ham(hartree_mol_list, nexciton, WFN, fe, fv, debug=True)
        self.assertAlmostEqual(Etot, 0.0)
        occ_std = np.array([[0.0], [0.0], [0.0]])
        self.assertTrue(np.allclose(A_el, occ_std))

    def test_SCF_exact(self):

        nexciton = 1

        dmrg_mol_list = custom_mol_list(None, ph_phys_dim, dis=[Quantity(0), Quantity(0)])
        # DMRG calculation
        procedure = [[40, 0.4], [40, 0.2], [40, 0.1], [40, 0], [40, 0]]
        mps, mpo = construct_mps_mpo_2(dmrg_mol_list, 40, nexciton, scheme=2)
        mps.optimize_config.procedure = procedure
        energy = optimize_mps(mps, mpo)
        dmrg_e = mps.expectation(mpo)

        # print occupation
        dmrg_occ = []
        for i in [0, 1, 2]:
            mpo = Mpo.onsite(dmrg_mol_list, "a^\dagger a", dipole=False, mol_idx_set={i})
            dmrg_occ.append(mps.expectation(mpo))
        print("dmrg_occ", dmrg_occ)

        hartree_mol_list = custom_mol_list(None, ph_phys_dim, dis=[Quantity(0), Quantity(0)], hartree=True)
        WFN, Etot = tdh.SCF(hartree_mol_list, nexciton)
        self.assertAlmostEqual(Etot, dmrg_e)

        fe, fv = 1, 6
        HAM, Etot, A_el = tdh.construct_H_Ham(hartree_mol_list, nexciton, WFN, fe, fv, debug=True)
        self.assertAlmostEqual(Etot, dmrg_e)
        self.assertTrue(np.allclose(A_el.flatten(), dmrg_occ))


    def test_TDH_ZT_emi(self):

        # disable tons of logging information
        log.init_log(logging.WARNING)
        nsteps = 3000 - 1
        dt = 30.0

        ls = tdh.LinearSpectra("emi", hartree_mol_list, prop_method="unitary")
        ls.evolve(dt, nsteps)
        with open("TDH_ZT_emi_prop1.npy", 'rb') as f:
            TDH_ZT_emi_prop1_std = np.load(f)
        self.assertTrue(np.allclose(ls.autocorr, TDH_ZT_emi_prop1_std))

        ls = tdh.LinearSpectra("emi", hartree_mol_list, prop_method="C_RK4")
        ls.evolve(dt, nsteps)
        with open("TDH_ZT_emi_RK4.npy", 'rb') as f:
            TDH_ZT_emi_RK4_std = np.load(f)
        self.assertTrue(np.allclose(ls.autocorr, TDH_ZT_emi_RK4_std))


    def test_TDH_ZT_abs(self):

        log.init_log(logging.WARNING)
        nsteps = 300 - 1
        dt = 10.0

        E_offset = -2.28614053 / constant.au2ev

        ls = tdh.LinearSpectra("abs", hartree_mol_list, E_offset=E_offset, prop_method="unitary")
        ls.evolve(dt, nsteps)
        with open("TDH_ZT_abs_prop1.npy", 'rb') as f:
            TDH_ZT_abs_prop1_std = np.load(f)
        self.assertTrue(np.allclose(ls.autocorr, TDH_ZT_abs_prop1_std))

        ls = tdh.LinearSpectra("abs", hartree_mol_list, E_offset=E_offset, prop_method="C_RK4")
        ls.evolve(dt, nsteps)
        with open("TDH_ZT_abs_RK4.npy", 'rb') as f:
            TDH_ZT_abs_RK4_std = np.load(f)
        self.assertTrue(np.allclose(ls.autocorr, TDH_ZT_abs_RK4_std))

    def test_1mol_ZTabs(self):
        log.init_log(logging.WARNING)
        nmols = 1

        mol_list = MolList([Mol(elocalex, hartree_ph_list, dipole_abs) for i in range(nmols)], np.zeros([1, 1]))

        nsteps = 1000 - 1
        dt = 30.0

        E_offset = -mol_list[0].elocalex - mol_list[0].hartree_e0

        ls = tdh.LinearSpectra("abs", mol_list, E_offset=E_offset, prop_method="unitary")
        ls.evolve(dt, nsteps)

        with open("1mol_ZTabs.npy", 'rb') as f:
            mol1_ZTabs_std = np.load(f)

        self.assertTrue(np.allclose(ls.autocorr, mol1_ZTabs_std))


if __name__ == "__main__":
    print("Test TDH")
    unittest.main()
