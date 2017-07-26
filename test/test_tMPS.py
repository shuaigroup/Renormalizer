# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import unittest
from ephMPS import tMPS
from parameter import *
from ephMPS import MPSsolver
from ephMPS import constant
from ephMPS.lib import mps as mpslib
from ddt import ddt, data

@ddt
class Test_tMPS(unittest.TestCase):
    
    def test_ZeroExactEmi(self):
        nexciton = 1
        procedure = [[10,0.4],[20,0.2],[30,0.1],[40,0],[40,0]]
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond = MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        
        MPSsolver.optimization(iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond,\
                nexciton, procedure, method="2site")
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        
        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond, "a", dipole=True)
        nsteps = 3000
        dt = 30.0
        autocorr = tMPS.ZeroTExactEmi(mol, pbond, iMPS, dipoleMPO, nsteps, dt)
        autocorr = np.array(autocorr)
        with open("std_data/tMPS/ZeroExactEmi.npy", 'rb') as f:
            ZeroExactEmi_std = np.load(f)
        self.assertTrue(np.allclose(autocorr,ZeroExactEmi_std,rtol=1e-4))


    @data([1,"svd"],[2,"svd"],[1,"variational"],[2,"variational"])
    def test_ZeroTcorr(self,value):
        nexciton = 0
        procedure = [[1,0],[1,0],[1,0]]
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond = \
        MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)

        MPSsolver.optimization(iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond,\
                        nexciton, procedure, method="2site")
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond,
                "a^\dagger", dipole=True)
        iMPS = mpslib.MPSdtype_convert(iMPS)
        
        nsteps = 100
        dt = 30.0

        autocorr = tMPS.ZeroTCorr(iMPS, HMPO, dipoleMPO, nsteps, dt, ephtable,
                thresh=1.0e-3, cleanexciton=1-nexciton, algorithm=value[0],
                compress_method=value[1])
        autocorr = np.array(autocorr)

        with open("std_data/tMPS/""ZeroTabs_"+str(value[0])+str(value[1])+".npy", 'rb') as f:
            ZeroTabs_std = np.load(f)
        self.assertTrue(np.allclose(autocorr,ZeroTabs_std,rtol=1e-4))
    

    @data([1,"svd"],[2,"svd"],[1,"variational"],[2,"variational"])
    def test_FiniteT_spectra_emi(self,value):
        nexciton = 1
        procedure = [[10,0.4],[20,0.2],[30,0.1],[40,0],[40,0]]
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond = MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        
        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond, "a", dipole=True)
        nsteps = 30
        dt = 30.0
        EXMPO = tMPS.Max_Entangled_EX_MPO(mol, pbond, norm=True)
        EXMPO = mpslib.MPSdtype_convert(EXMPO)
    
        insteps = 50
        autocorr = tMPS.FiniteT_spectra("emi", mol, pbond, EXMPO, HMPO,
                dipoleMPO, nsteps, dt, ephtable, insteps, thresh=1.0e-3,
                temperature=298, algorithm=value[0], compress_method=value[1])
        
        autocorr = np.array(autocorr)
        
        with open("std_data/tMPS/TTemi_"+str(value[0])+str(value[1])+".npy", 'rb') as f:
            TTemi_std = np.load(f)
        self.assertTrue(np.allclose(autocorr,TTemi_std[0:nsteps],rtol=1e-4))


    @data([1,"svd"],[1,"variational"])
    def test_FiniteT_spectra_abs(self,value):
        nexciton = 0
        procedure = [[1,0],[1,0],[1,0]]
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond = MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        
        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond,
                "a^\dagger", dipole=True)
        GSMPS, GSMPSdim = tMPS.Max_Entangled_GS_MPS(mol, pbond)
        GSMPO = tMPS.hilbert_to_liouville(GSMPS)
        GSMPO = mpslib.MPSdtype_convert(GSMPO)

        nsteps = 50
        dt = 30.0
        autocorr = tMPS.FiniteT_spectra("abs", mol, pbond, GSMPO, HMPO,
                dipoleMPO, nsteps, dt, ephtable, thresh=1.0e-3,
                temperature=298, algorithm=value[0], compress_method=value[1])
        
        autocorr = np.array(autocorr)

        with open("std_data/tMPS/TTabs_"+str(value[1]+".npy"), 'rb') as f:
            TTabs_std = np.load(f)

        self.assertTrue(np.allclose(autocorr,TTabs_std[0:nsteps],rtol=1e-4))


    @data(["svd"],["variational"])
    def test_FiniteT_emi(self,value):
        nexciton = 1
        procedure = [[10,0.4],[20,0.2],[30,0.1],[40,0],[40,0]]
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond = MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        
        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond, "a", dipole=True)
        nsteps = 30
        dt = 30.0
        EXMPO = tMPS.Max_Entangled_EX_MPO(mol, pbond, norm=True)
        EXMPO = mpslib.MPSdtype_convert(EXMPO)
    
        insteps = 50
        autocorr = tMPS.FiniteT_emi(mol, pbond, EXMPO, HMPO,
                dipoleMPO, nsteps, dt, ephtable, insteps, thresh=1.0e-3,
                temperature=298, compress_method=value[0])
        
        autocorr = np.array(autocorr)
        
        with open("std_data/tMPS/TTemi_"+str(1)+str(value[0])+".npy", 'rb') as f:
            TTemi_std = np.load(f)
        self.assertTrue(np.allclose(autocorr,TTemi_std[0:nsteps],rtol=1e-4))


    @data(["svd"],["variational"])
    def test_FiniteT_abs(self,value):
        nexciton = 0
        procedure = [[1,0],[1,0],[1,0]]
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond = MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        
        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond,
                "a^\dagger", dipole=True)
        GSMPS, GSMPSdim = tMPS.Max_Entangled_GS_MPS(mol, pbond)
        GSMPO = tMPS.hilbert_to_liouville(GSMPS)
        GSMPO = mpslib.MPSdtype_convert(GSMPO)

        nsteps = 50
        dt = 30.0
        autocorr = tMPS.FiniteT_abs(mol, pbond, GSMPO, HMPO,
                dipoleMPO, nsteps, dt, ephtable, thresh=1.0e-3,
                temperature=298, compress_method=value[0])
        
        autocorr = np.array(autocorr)

        with open("std_data/tMPS/TTabs_"+str(value[0]+".npy"), 'rb') as f:
            TTabs_std = np.load(f)

        self.assertTrue(np.allclose(autocorr,TTabs_std[0:nsteps],rtol=1e-4))


if __name__ == "__main__":
    print("Test tMPS")
    unittest.main()
