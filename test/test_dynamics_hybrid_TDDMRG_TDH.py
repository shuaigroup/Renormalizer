# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
import numpy as np
import unittest
from ddt import ddt, data
import parameter_PBI
from ephMPS import TDH
from ephMPS import MPSsolver
from ephMPS import hybrid_TDDMRG_TDH
from ephMPS import tMPS
from ephMPS import RK
from ephMPS.lib import mps as mpslib

@ddt
class Test_dynamics_hybrid_TDDMRG_TDH(unittest.TestCase):
    

    @data(\
            [10],\
            [5])
    def test_ZT_dynamics_hybrid_TDDMRG_TDH(self,value):
        
        mol, J = parameter_PBI.construct_mol(4,nphs=value[0])
        TDH.construct_Ham_vib(mol, hybrid=True)
        nexciton = 0
        dmrg_procedure = [[20,0.5],[20,0.3],[10,0.2],[5,0],[1,0]]
        MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
            MPSsolver.construct_MPS_MPO_2(mol, J, dmrg_procedure[0][0], nexciton)
        
        HMPO_init = [MPO, MPOQN, MPOQNidx, MPOQNtot]
        
        MPS, MPSQN, WFN, Etot = hybrid_TDDMRG_TDH.hybrid_DMRG_H_SCF(mol, J, \
                nexciton, dmrg_procedure, 20, DMRGthresh=1e-7, Hthresh=1e-7)
        
        iMPS = [MPS, MPSQN, len(MPS)-1, 0]
        QNargs = [ephtable, False]
        
        dipoleMPO, dipoleMPOdim = MPSsolver.construct_onsiteMPO(mol, pbond, "a^\dagger",\
                QNargs=QNargs, sitelist=[0])
        
        
        iMPS = mpslib.mapply(dipoleMPO, iMPS, QNargs=QNargs)
        norm = mpslib.norm(iMPS, QNargs=QNargs)
        WFN[-1] *= norm
        iMPS = mpslib.scale(iMPS,1./norm,QNargs=QNargs)
        iMPS = mpslib.MPSdtype_convert(iMPS, QNargs=QNargs)
        WFN = [wfn.astype(np.complex128) for wfn in WFN[:-1]]+[WFN[-1]]
        
        nsteps = 30
        dt = 30.0
        
        MPOs = []
        for imol in xrange(len(mol)):
            dipoleMPO, dipoleMPOdim = MPSsolver.construct_onsiteMPO(mol, pbond, "a^\dagger a",\
                QNargs=QNargs, sitelist=[imol])
            MPOs.append(dipoleMPO)
        
        rk = RK.Runge_Kutta(method="C_RK4")
        #rk = RK.Runge_Kutta(method="RKF45", rtol=1e-3, adaptive=True)
        setup = tMPS.prop_setup(rk)

        tlist, data = hybrid_TDDMRG_TDH.dynamics_hybrid_TDDMRG_TDH(setup, mol, \
                J, HMPO_init, iMPS, \
                WFN, nsteps, dt, ephtable,thresh=1e-3, QNargs=QNargs, property_MPOs=MPOs)
        
        with open("std_data/hybrid_TDDMRG_TDH/ZT_occ"+str(value[0])+".npy", 'rb') as f:
            std = np.load(f)
        self.assertTrue(np.allclose(data,std))


    @data(\
            [10],\
            [5])
    def test_FT_dynamics_hybrid_TDDMRG_TDH(self,value):
        
        mol, J = parameter_PBI.construct_mol(4,nphs=value[0])
        TDH.construct_Ham_vib(mol, hybrid=True)
        nexciton = 0
        dmrg_procedure = [[20,0.5],[20,0.3],[10,0.2],[5,0],[1,0]]
        MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
            MPSsolver.construct_MPS_MPO_2(mol, J, dmrg_procedure[0][0], nexciton)
        
        HMPO_init = [MPO, MPOQN, MPOQNidx, MPOQNtot]
        QNargs = [ephtable, False]
        T = 2000.
        insteps = 1

        rk = RK.Runge_Kutta(method="C_RK4")
        setup = tMPS.prop_setup(rk)
        
        iMPS, WFN = hybrid_TDDMRG_TDH.FT_DM_hybrid_TDDMRG_TDH(rk, mol, J,\
                HMPO_init, nexciton, T, \
                insteps, pbond, ephtable, thresh=1e-3, cleanexciton=nexciton,\
                QNargs=QNargs, space="GS")
    
        dipoleMPO, dipoleMPOdim = MPSsolver.construct_onsiteMPO(mol, pbond, "a^\dagger",\
                QNargs=QNargs, sitelist=[0])
        
        iMPS = mpslib.mapply(dipoleMPO, iMPS, QNargs=QNargs)
        norm = mpslib.norm(iMPS, QNargs=QNargs)
        WFN[-1] *= norm
        iMPS = mpslib.scale(iMPS,1./norm,QNargs=QNargs)
        iMPS = mpslib.MPSdtype_convert(iMPS, QNargs=QNargs)
        WFN = [wfn.astype(np.complex128) for wfn in WFN[:-1]]+[WFN[-1]]
        
        nsteps = 90
        dt = 10.0
        
        MPOs = []
        for imol in xrange(len(mol)):
            dipoleMPO, dipoleMPOdim = MPSsolver.construct_onsiteMPO(mol, pbond, "a^\dagger a",\
                QNargs=QNargs, sitelist=[imol])
            MPOs.append(dipoleMPO)
        
        #rk = RK.Runge_Kutta(method="RKF45", rtol=1e-3, adaptive=True)
        #setup = tMPS.prop_setup(rk)
        
        tlist, data = hybrid_TDDMRG_TDH.dynamics_hybrid_TDDMRG_TDH(setup, mol, \
                J, HMPO_init, iMPS, \
                WFN, nsteps, dt, ephtable,thresh=1e-3, QNargs=QNargs, property_MPOs=MPOs)
        
        with open("std_data/hybrid_TDDMRG_TDH/FT_occ"+str(value[0])+".npy", 'rb') as f:
            std = np.load(f)
        self.assertTrue(np.allclose(data,std))


if __name__ == "__main__":
    print("Test dynamics_hybrid_TDDMRG_TDH")
    unittest.main()
