# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import unittest
from ephMPS import tMPS
from ephMPS.test.parameter import *
from ephMPS import MPSsolver
from ephMPS import constant
from ephMPS.lib import mps as mpslib
from ddt import ddt, data
import os
import glob

@ddt
class Test_tMPS_TDVP(unittest.TestCase):
    
    @data(\
           ["TDVP_PS",2, 15.0, 1e-2],\
           ["TDVP_MCTDHnew",2, 2.0, 1e-2])
    def test_ZeroTcorr_TDVP(self,value):
        
        nexciton = 0
        procedure = [[50,0],[50,0],[50,0]]
        
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond = \
        MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)

        MPSsolver.optimization(iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond,\
                        nexciton, procedure, method="2site")
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        
        dipoleMPO, dipoleMPOdim = MPSsolver.construct_onsiteMPO(mol, pbond,
                "a^\dagger", dipole=True)
        iMPS = mpslib.MPSdtype_convert(iMPS)
        
        nsteps = 200
        dt = value[2]

        for f in glob.glob("TDVP_PS*.npy"):
            os.remove(f)

        autocorr = tMPS.ZeroTCorr(iMPS, HMPO, dipoleMPO, nsteps, dt, ephtable,
                thresh=1.0e-7, cleanexciton=1-nexciton, algorithm=value[1],
                compress_method="variational", scheme=value[0])

        with open("/std_data/tMPS/ZeroTabs_"+value[0]+".npy", 'rb') as f:
            ZeroTabs_std = np.load(f)
        self.assertTrue(np.allclose(autocorr,ZeroTabs_std[:len(autocorr)],rtol=value[3]))
    
    @data(\
            ["TDVP_MCTDHnew",191,2.0,1e-2,"_TDVP_MCTDHnew"],\
            ["TDVP_PS",30,30,1e-2,""])
    def test_FiniteT_spectra_emi(self,value):
        nexciton = 1
        procedure = [[10,0.4],[20,0.2],[30,0.1],[40,0],[40,0]]
        
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond = \
            MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        
        
        dipoleMPO, dipoleMPOdim = MPSsolver.construct_onsiteMPO(mol, pbond, "a", dipole=True)
        nsteps = value[1]
        dt = value[2]
        EXMPO, EXMPOdim = tMPS.Max_Entangled_EX_MPO(mol, pbond, norm=True)
        EXMPO = mpslib.MPSdtype_convert(EXMPO)

        for f in glob.glob("TDVP_PS*.npy"):
            os.remove(f)
        
        insteps = 50
        autocorr = tMPS.FiniteT_spectra("emi", mol, pbond, EXMPO, HMPO,
                dipoleMPO, nsteps, dt, ephtable, insteps, thresh=1.0e-3,
                temperature=298, compress_method="variational", scheme=value[0])
        
        with open("/std_data/tMPS/TTemi_2svd"+value[4]+".npy", 'rb') as f:
            TTemi_std = np.load(f)
        self.assertTrue(np.allclose(autocorr,TTemi_std[0:nsteps],rtol=value[3]))

if __name__ == "__main__":
    print("Test tMPS_TDVP")
    unittest.main()
