# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import copy
import unittest
from ddt import ddt, data
from parameter import *
from ephMPS import TDH
from ephMPS import MPSsolver
from ephMPS.lib import mps as mpslib
from ephMPS import tMPS
from ephMPS import constant


@ddt
class Test_TDH(unittest.TestCase):
    
    def test_SCF(self):
        
        #  EX   
        nexciton= 1
        WFN, Etot = TDH.SCF(mol, J, nexciton)
        self.assertAlmostEqual(Etot, 0.0843103276663)
        
        fe, fv = 1, 6
        HAM, Etot, A_el = TDH.construct_H_Ham(mol, J, nexciton, WFN, fe, fv, debug=True)
        self.assertAlmostEqual(Etot, 0.0843103276663)
        occ_std = np.array([[0.20196397], [0.35322702],[0.444809]])
        self.assertTrue(np.allclose(A_el, occ_std))
    
        # GS
        nexciton= 0
        WFN, Etot = TDH.SCF(mol, J, nexciton)
        self.assertAlmostEqual(Etot, 0.0)
        
        fe, fv = 1, 6
        HAM, Etot, A_el = TDH.construct_H_Ham(mol, J, nexciton, WFN, fe, fv, debug=True)
        self.assertAlmostEqual(Etot, 0.0)
        occ_std = np.array([[0.0], [0.0],[0.0]])
        self.assertTrue(np.allclose(A_el, occ_std))
    

    def test_SCF_exact(self):
        
        nexciton= 1
        D_value = np.array([0.0, 0.0])
        mol = construct_mol(nlevels, D_value=D_value)
        # DMRG calculation
        procedure = [[40,0.4],[40,0.2],[40,0.1],[40,0],[40,0]]
        
        MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
            MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        energy = MPSsolver.optimization(MPS, MPSdim, MPSQN, MPO, MPOdim, 
                ephtable, pbond, nexciton, procedure, method="2site")
        dmrg_e = mpslib.dot(MPS, mpslib.mapply(MPO, MPS))
        
        # print occupation
        dmrg_occ = []
        for i in [0,1,2]:
            MPO, MPOdim = tMPS.construct_onsiteMPO(mol,pbond,"a^\dagger a",dipole=False,sitelist=[i])
            dmrg_occ.append(mpslib.dot(MPS, mpslib.mapply(MPO, MPS)))
        print "dmrg_occ", dmrg_occ

        WFN, Etot = TDH.SCF(mol, J, nexciton)
        self.assertAlmostEqual(Etot, dmrg_e)
        
        fe, fv = 1, 6
        HAM, Etot, A_el = TDH.construct_H_Ham(mol, J, nexciton, WFN, fe, fv, debug=True)
        self.assertAlmostEqual(Etot, dmrg_e)
        self.assertTrue(np.allclose(A_el.flatten(), dmrg_occ))

    
    def test_TDH_ZT_emi(self):
        
        nexciton= 1
        WFN, Etot = TDH.SCF(mol, J, nexciton)

        nsteps = 3000
        dt = 30.0
        fe, fv = 1, 6

        WFN = [wfn.astype(np.complex128) for wfn in WFN[:-1]]+[WFN[-1]]
        
        WFN0 = copy.deepcopy(WFN)
        autocorr = TDH.linear_spectra("emi", mol, J, nexciton, WFN0, dt, nsteps, fe, fv, prop_method="unitary")
        with open("std_data/TDH/TDH_ZT_emi_prop1.npy", 'rb') as f:
            TDH_ZT_emi_prop1_std = np.load(f)
        self.assertTrue(np.allclose(autocorr,TDH_ZT_emi_prop1_std))
        
        WFN0 = copy.deepcopy(WFN)
        autocorr = TDH.linear_spectra("emi", mol, J, nexciton, WFN0, dt, nsteps, fe, fv, prop_method="C_RK4")
        with open("std_data/TDH/TDH_ZT_emi_RK4.npy", 'rb') as f:
            TDH_ZT_emi_RK4_std = np.load(f)
        self.assertTrue(np.allclose(autocorr,TDH_ZT_emi_RK4_std))


    def test_TDH_ZT_abs(self):
        
        nexciton= 0
        WFN, Etot = TDH.SCF(mol, J, nexciton)

        nsteps = 300
        dt = 10.0
        fe, fv = 1, 6

        WFN = [wfn.astype(np.complex128) for wfn in WFN[:-1]]+[WFN[-1]]
        E_offset = -2.28614053/constant.au2ev

        WFN0 = copy.deepcopy(WFN)
        autocorr = TDH.linear_spectra("abs", mol, J, nexciton, WFN0, dt, nsteps, fe, fv, E_offset=E_offset, prop_method="unitary")
        with open("std_data/TDH/TDH_ZT_abs_prop1.npy", 'rb') as f:
            TDH_ZT_abs_prop1_std = np.load(f)
        self.assertTrue(np.allclose(autocorr,TDH_ZT_abs_prop1_std))
        
        WFN0 = copy.deepcopy(WFN)
        autocorr = TDH.linear_spectra("abs", mol, J, nexciton, WFN0, dt, nsteps, fe, fv, E_offset=E_offset, prop_method="C_RK4")
        with open("std_data/TDH/TDH_ZT_abs_RK4.npy", 'rb') as f:
            TDH_ZT_abs_RK4_std = np.load(f)
        self.assertTrue(np.allclose(autocorr,TDH_ZT_abs_RK4_std))
    

    def test_1mol_ZTabs(self):
        nmols = 1
        J = np.zeros([1,1])
        
        mol = []
        for imol in xrange(nmols):
            mol_local = obj.Mol(elocalex, nphs, dipole_abs)
            mol_local.create_ph(phinfo)
            mol.append(mol_local)
        
        nexciton = 0
        
        # TDH
        WFN, Etot = TDH.SCF(mol, J, nexciton)
        print "SCF Etot", Etot
        nsteps = 1000
        dt = 30.0
        fe, fv = 1, 2

        WFN = [wfn.astype(np.complex128) for wfn in WFN[:-1]]+[WFN[-1]]
        E_offset = -mol[0].elocalex-mol[0].e0

        WFN0 = copy.deepcopy(WFN)
        autocorr = TDH.linear_spectra("abs", mol, J, nexciton, WFN0, dt, nsteps, fe, fv, E_offset=E_offset, prop_method="unitary")
        
        with open("std_data/tMPS/1mol_ZTabs.npy", 'rb') as f:
            mol1_ZTabs_std = np.load(f)

        self.assertTrue(np.allclose(autocorr,mol1_ZTabs_std))


if __name__ == "__main__":
    print("Test TDH")
    unittest.main()
