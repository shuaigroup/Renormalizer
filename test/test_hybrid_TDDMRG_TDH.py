# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
import numpy as np
import unittest
import copy
from ddt import ddt, data
from ephMPS import hybrid_TDDMRG_TDH
from ephMPS import MPSsolver
from ephMPS import tMPS
from ephMPS.lib import mps as mpslib
from ephMPS.constant import *
from ephMPS import TDH
from ephMPS import RK 

from parameter_hybrid import *
mol_hybrid = mol
from parameter import *
mol_pure = mol

dmrg_procedure = [[10,0.4],[20,0.2],[30,0.1],[40,0],[40,0]]

@ddt
class Test_hybrid_TDDMRG_TDH(unittest.TestCase):
    
    @data(\
            [mol_hybrid,0.084015672468],\
            [mol_pure,0.08401411562239858])
    def test_hybrid_DMRG_H_SCF(self,value):
        
        TDH.construct_Ham_vib(value[0], hybrid=True)
        nexciton = 1
        niterations = 20
        MPS, MPSQN, WFN, Etot = hybrid_TDDMRG_TDH.hybrid_DMRG_H_SCF(value[0], J, \
                nexciton, dmrg_procedure, 20, DMRGthresh=1e-5, Hthresh=1e-5)
        print "Etot", Etot
        self.assertAlmostEqual(Etot, value[1])
        
        nexciton = 0
        niterations = 20
        MPS, MPSQN, WFN, Etot = hybrid_TDDMRG_TDH.hybrid_DMRG_H_SCF(value[0], J, \
                nexciton, dmrg_procedure, 20, DMRGthresh=1e-5, Hthresh=1e-5)
        print "Etot", Etot
        self.assertAlmostEqual(Etot, 0.0)
 

    @data(\
            [mol_hybrid,"std_data/hybrid_TDDMRG_TDH/hybrid_ZTabs.npy",1e-5],\
            [mol_pure,"std_data/tMPS/ZeroTabs_2svd.npy",1e-2])
    def test_ZeroTcorr_hybrid_TDDMRG_TDH_abs(self,value):
        
        TDH.construct_Ham_vib(value[0], hybrid=True)
        
        nexciton = 0
        niterations = 20
        
        MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
            MPSsolver.construct_MPS_MPO_2(value[0], J, dmrg_procedure[0][0], nexciton)

        MPS, MPSQN, WFN, Etot = hybrid_TDDMRG_TDH.hybrid_DMRG_H_SCF(value[0], J, \
                nexciton, dmrg_procedure, 20, DMRGthresh=1e-5, Hthresh=1e-5)
        iMPS = [MPS, MPSQN, len(MPS)-1, 0]
        QNargs = [ephtable, False]
        dipoleMPO, dipoleMPOdim = MPSsolver.construct_onsiteMPO(value[0], pbond, "a^\dagger", dipole=True, QNargs=QNargs)
        
        nsteps = 100
        dt = 30.0
        iMPS = mpslib.MPSdtype_convert(iMPS, QNargs=QNargs)
        WFN = [wfn.astype(np.complex128) for wfn in WFN[:-1]]+[WFN[-1]]

        rk = RK.Runge_Kutta("C_RK4")
        setup = tMPS.prop_setup(rk)

        autocorr = hybrid_TDDMRG_TDH.ZeroTcorr_hybrid_TDDMRG_TDH(setup, value[0], J, iMPS, dipoleMPO, \
                WFN, nsteps, dt, ephtable,thresh=1e-3, E_offset=-2.28614053/constant.au2ev, QNargs=QNargs)
        with open(value[1], 'rb') as f:
            hybrid_ZTabs_std = np.load(f)
        self.assertTrue(np.allclose(autocorr,hybrid_ZTabs_std,rtol=value[2]))
    

    @data(\
            [mol_hybrid,"std_data/hybrid_TDDMRG_TDH/hybrid_ZTemi_prop.npy",1e-5],\
            [mol_pure,"std_data/tMPS/ZeroExactEmi.npy",1e-3])
    def test_ZeroTcorr_hybrid_TDDMRG_TDH_emi(self,value):
        
        TDH.construct_Ham_vib(value[0], hybrid=True)
        nexciton = 1
        niterations = 20
        
        MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
            MPSsolver.construct_MPS_MPO_2(value[0], J, dmrg_procedure[0][0], nexciton)

        MPS, MPSQN, WFN, Etot = hybrid_TDDMRG_TDH.hybrid_DMRG_H_SCF(value[0], J, \
                nexciton, dmrg_procedure, 20, DMRGthresh=1e-5, Hthresh=1e-5)
        iMPS = [MPS, MPSQN, len(MPS)-1, 1]
        QNargs = [ephtable, False]
        dipoleMPO, dipoleMPOdim = MPSsolver.construct_onsiteMPO(value[0], pbond, "a", dipole=True, QNargs=QNargs)
        
        nsteps = 1000
        dt = 30.0
        iMPS = mpslib.MPSdtype_convert(iMPS, QNargs=QNargs)
        WFN = [wfn.astype(np.complex128) for wfn in WFN[:-1]]+[WFN[-1]]
        
        rk = RK.Runge_Kutta("C_RK4")
        setup = tMPS.prop_setup(rk)
        
        autocorr = hybrid_TDDMRG_TDH.ZeroTcorr_hybrid_TDDMRG_TDH(setup, value[0], J, iMPS, dipoleMPO, \
                WFN, nsteps, dt, ephtable,thresh=1e-3, QNargs=QNargs)
        with open(value[1], 'rb') as f:
            hybrid_ZTemi_prop_std = np.load(f)[:nsteps]
        self.assertTrue(np.allclose(autocorr,hybrid_ZTemi_prop_std,rtol=value[2]))


    @data(\
            [mol_hybrid,"std_data/hybrid_TDDMRG_TDH/hybrid_ZTemi_exact.npy",1e-5],\
            [mol_pure,"std_data/tMPS/ZeroExactEmi.npy",1e-3])
    def test_Exact_Spectra_hybrid_TDDMRG_TDH(self,value):
        
        TDH.construct_Ham_vib(value[0], hybrid=True)
        nexciton = 1
        niterations = 20
        
        MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
            MPSsolver.construct_MPS_MPO_2(value[0], J, dmrg_procedure[0][0], nexciton)

        MPS, MPSQN, WFN, Etot = hybrid_TDDMRG_TDH.hybrid_DMRG_H_SCF(value[0], J, \
                nexciton, dmrg_procedure, 20, DMRGthresh=1e-5, Hthresh=1e-5)
        dipoleMPO, dipoleMPOdim = MPSsolver.construct_onsiteMPO(value[0], pbond, "a", dipole=True)
        
        nsteps = 3000
        dt = 30.0
        MPS = mpslib.MPSdtype_convert(MPS)
        WFN = [wfn.astype(np.complex128) for wfn in WFN[:-1]]+[WFN[-1]]
        
        autocorr = hybrid_TDDMRG_TDH.Exact_Spectra_hybrid_TDDMRG_TDH("emi", value[0], J, MPS, \
                dipoleMPO, WFN, nsteps, dt)
        with open(value[1], 'rb') as f:
            hybrid_ZTemi_exact_std = np.load(f)
        self.assertTrue(np.allclose(autocorr,hybrid_ZTemi_exact_std,rtol=value[2]))

    
    @data(\
            ["pure"],\
            ["hybrid"])
    def test_1mol_Exact_Spectra_hybrid_TDDMRG_TDH(self,value):
        
        nmols = 1
        J = np.zeros([1,1])
        
        if value[0] == "pure":
            nphs = 2
            phinfo = [list(a) for a in zip(omega, D, nlevels)]
            mol = []
            for imol in xrange(nmols):
                mol_local = obj.Mol(elocalex, nphs, dipole_abs)
                mol_local.create_ph(phinfo)
                mol.append(mol_local)
        
        elif value[0] == "hybrid":
            nphs = 1
            nphs_hybrid = 1
            phinfo_hybrid = [list(a) for a in zip(omega[:nphs], D[:nphs], nlevels[:nphs])]
            phinfo = [list(a) for a in zip(omega[nphs:], D[nphs:], nlevels[nphs:])]
            
            mol = []
            for imol in xrange(nmols):
                mol_local = obj.Mol(elocalex, nphs, dipole_abs, nphs_hybrid=nphs_hybrid)
                mol_local.create_ph(phinfo)
                mol_local.create_ph(phinfo_hybrid, phtype="hybrid")
                mol.append(mol_local)
        
        TDH.construct_Ham_vib(mol, hybrid=True)
        nexciton = 0
        dmrg_procedure = [[10,0.4],[20,0.2],[30,0.1],[40,0],[40,0]]
        
        MPS, MPSdim, MPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond \
                = MPSsolver.construct_MPS_MPO_2(mol, J, dmrg_procedure[0][0], nexciton)
        
        MPS, MPSQN, WFN, Etot = hybrid_TDDMRG_TDH.hybrid_DMRG_H_SCF(mol, J, \
                nexciton, dmrg_procedure, 20, DMRGthresh=1e-5, Hthresh=1e-5)
        dipoleMPO, dipoleMPOdim = MPSsolver.construct_onsiteMPO(mol, pbond, "a^\dagger", dipole=True)
        
        nsteps = 1000
        dt = 30.0
        MPS = mpslib.MPSdtype_convert(MPS)
        WFN = [wfn.astype(np.complex128) for wfn in WFN[:-1]]+[WFN[-1]]
        
        E_offset = -mol[0].elocalex-mol[0].e0-mol[0].e0_hybrid
        autocorr = hybrid_TDDMRG_TDH.Exact_Spectra_hybrid_TDDMRG_TDH("abs", mol, J, MPS, \
                dipoleMPO, WFN, nsteps, dt, E_offset=E_offset)
        
        with open("std_data/tMPS/1mol_ZTabs.npy", 'rb') as f:
            mol1_ZTabs_std = np.load(f)

        self.assertTrue(np.allclose(autocorr,mol1_ZTabs_std,rtol=1e-3))


if __name__ == "__main__":
    print("Test hybrid_TDDMRG_TDH")
    unittest.main()
