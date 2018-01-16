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
class Test_tMPS_eiHt(unittest.TestCase):
    
    @data(\
            [1,"svd",True,[[4,4]],1e-2],\
            [2,"svd",True,[[4,4]],1e-2],\
            [1,"svd",None,[[4,4]],1e-2],\
            [2,"svd",None,[[4,4]],1e-2],\
            [1,"svd",True,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [2,"svd",True,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [1,"svd",None,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [2,"svd",None,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2])
    def test_ZeroTcorr_eiHt(self,value):
        
        print "data", value
        nexciton = 0
        procedure = [[1,0],[1,0],[1,0]]
        
        mol = construct_mol(*value[3])
        
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond = \
        MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)

        MPSsolver.optimization(iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond,\
                        nexciton, procedure, method="2site")
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        
        if value[2] != None:
            iMPS = [iMPS, iMPSQN, len(iMPS)-1, 0]
            QNargs = [ephtable, False]
            HMPO = [HMPO, HMPOQN, HMPOQNidx, HMPOQNtot]
        else:
            QNargs = None

        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond,
                "a^\dagger", dipole=True, QNargs=QNargs)
        iMPS = mpslib.MPSdtype_convert(iMPS, QNargs=QNargs)
        
        nsteps = 100
        dt = 5.0
        
        autocorr0 = tMPS.ZeroTCorr(iMPS, HMPO, dipoleMPO, nsteps, dt, ephtable,
                thresh=1.0e-3, cleanexciton=1-nexciton, algorithm=value[0],
                compress_method=value[1], QNargs=QNargs, approxeiHt=None)
        autocorr0 = np.array(autocorr0)
        
        autocorr1 = tMPS.ZeroTCorr(iMPS, HMPO, dipoleMPO, nsteps, dt, ephtable,
                thresh=1.0e-3, cleanexciton=1-nexciton, algorithm=value[0],
                compress_method=value[1], QNargs=QNargs, approxeiHt=1e-6)
        autocorr1 = np.array(autocorr1)
        
        self.assertTrue(np.allclose(autocorr0,autocorr1,rtol=value[4]))

    @data(\
            [2,"svd",True,[[4,4]],1e-2],\
            [2,"svd",None,[[4,4]],1e-2],\
            [2,"svd",True,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [2,"svd",None,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2])
    def test_FiniteT_spectra_emi_eiHt(self,value):
        print "data", value
        nexciton = 1
        procedure = [[10,0.4],[20,0.2],[30,0.1],[40,0],[40,0]]
        mol = construct_mol(*value[3])
        
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond = \
            MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        
        if value[2] != None:
            QNargs = [ephtable, False]
            HMPO = [HMPO, HMPOQN, HMPOQNidx, HMPOQNtot]
        else:
            QNargs = None
        
        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond,
                "a", dipole=True, QNargs=QNargs)
        
        nsteps = 100
        dt = 5.0
       
        EXMPO, EXMPOdim = tMPS.Max_Entangled_EX_MPO(mol, pbond, norm=True, QNargs=QNargs)
        EXMPO = mpslib.MPSdtype_convert(EXMPO, QNargs=QNargs)

        insteps = 200
        
        autocorr0 = tMPS.FiniteT_spectra("emi", mol, pbond, EXMPO, HMPO,
                dipoleMPO, nsteps, dt, ephtable, insteps, thresh=1.0e-3,
                temperature=298, algorithm=value[0], compress_method=value[1],
                QNargs=QNargs, approxeiHt=None)
        autocorr0 = np.array(autocorr0)
        
        autocorr1 = tMPS.FiniteT_spectra("emi", mol, pbond, EXMPO, HMPO,
                dipoleMPO, nsteps, dt, ephtable, insteps, thresh=1.0e-3,
                temperature=298, algorithm=value[0], compress_method=value[1],
                QNargs=QNargs, approxeiHt=1e-6)
        autocorr1 = np.array(autocorr1)
        
        self.assertTrue(np.allclose(autocorr0, autocorr1, rtol=value[4]))


    @data(\
            [2,"svd",True,[[4,4]],1e-2],\
            [2,"svd",None,[[4,4]],1e-2],\
            [2,"svd",True,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [2,"svd",None,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2])
    def test_FiniteT_spectra_abs_eiHt(self,value):
        print "data", value
        nexciton = 0
        procedure = [[1,0],[1,0],[1,0]]
        mol = construct_mol(*value[3])
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond = MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        
        if value[2] != None:
            QNargs = [ephtable, False]
            HMPO = [HMPO, HMPOQN, HMPOQNidx, HMPOQNtot]
        else:
            QNargs = None
        
        
        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond,
                "a^\dagger", dipole=True, QNargs=QNargs)
        GSMPS, GSMPSdim = tMPS.Max_Entangled_GS_MPS(mol, pbond, QNargs=QNargs)
        GSMPO = tMPS.hilbert_to_liouville(GSMPS, QNargs=QNargs)
        GSMPO = mpslib.MPSdtype_convert(GSMPO, QNargs=QNargs)

        nsteps = 100
        dt = 5.0
    
        autocorr0 = tMPS.FiniteT_spectra("abs", mol, pbond, GSMPO, HMPO,
                dipoleMPO, nsteps, dt, ephtable, thresh=1.0e-3,
                temperature=298, algorithm=value[0], compress_method=value[1],\
                QNargs=QNargs, approxeiHt=None)
        autocorr0 = np.array(autocorr0)

        autocorr1 = tMPS.FiniteT_spectra("abs", mol, pbond, GSMPO, HMPO,
                dipoleMPO, nsteps, dt, ephtable, thresh=1.0e-3,
                temperature=298, algorithm=value[0], compress_method=value[1],\
                QNargs=QNargs, approxeiHt=1e-6)
        autocorr1 = np.array(autocorr1)

        self.assertTrue(np.allclose(autocorr0,autocorr1,rtol=value[4]))

if __name__ == "__main__":
    print("Test tMPS_eiHt")
    unittest.main()
