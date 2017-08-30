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

    @data([[[4,4]], 1e-3],\
            [[[4,4],[2,2],[1.e-7,1.e-7]], 1e-3])
    def test_ZeroExactEmi(self, value):
        
        print "data", value
        nexciton = 1
        procedure = [[10,0.4],[20,0.2],[30,0.1],[40,0],[40,0]]
        
        mol = construct_mol(*value[0])
        
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond = MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        
        MPSsolver.optimization(iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond,\
                nexciton, procedure, method="2site")

        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        
        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond, "a", dipole=True)
        nsteps = 3000
        dt = 30.0
        temperature = 0
        autocorr = tMPS.Exact_Spectra("emi", mol, pbond, iMPS, dipoleMPO, \
                nsteps, dt, temperature)
        autocorr = np.array(autocorr)
        with open("std_data/tMPS/ZeroExactEmi.npy", 'rb') as f:
            ZeroExactEmi_std = np.load(f)
        
        self.assertTrue(np.allclose(autocorr,ZeroExactEmi_std,rtol=value[1]))


    @data(\
            [1,"svd",True,[[4,4]],1e-3],\
            [2,"svd",True,[[4,4]],1e-3],\
            [1,"svd",None,[[4,4]],1e-3],\
            [2,"svd",None,[[4,4]],1e-3],\
            [1,"variational",None,[[4,4]],1e-3],\
            [2,"variational",None,[[4,4]],1e-3],\
            [1,"svd",True,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [2,"svd",True,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [1,"svd",None,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [2,"svd",None,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [1,"variational",None,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [2,"variational",None,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2])
    def test_ZeroTcorr(self,value):
        
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
            QNargs = [ephtable]
            HMPO = [HMPO, HMPOQN, HMPOQNidx, HMPOQNtot]
        else:
            QNargs = None

        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond,
                "a^\dagger", dipole=True, QNargs=QNargs)
        iMPS = mpslib.MPSdtype_convert(iMPS, QNargs=QNargs)
        

        nsteps = 100
        dt = 30.0

        autocorr = tMPS.ZeroTCorr(iMPS, HMPO, dipoleMPO, nsteps, dt, ephtable,
                thresh=1.0e-3, cleanexciton=1-nexciton, algorithm=value[0],
                compress_method=value[1], QNargs=QNargs)
        autocorr = np.array(autocorr)

        with open("std_data/tMPS/""ZeroTabs_"+str(value[0])+str(value[1])+".npy", 'rb') as f:
            ZeroTabs_std = np.load(f)
        self.assertTrue(np.allclose(autocorr,ZeroTabs_std,rtol=value[4]))
    
    
    @data(\
            [1,"svd",True,[[4,4]],1e-3],\
            [2,"svd",True,[[4,4]],1e-3],\
            [1,"svd",None,[[4,4]],1e-3],\
            [2,"svd",None,[[4,4]],1e-3],\
            [1,"variational",None,[[4,4]],1e-3],\
            [2,"variational",None,[[4,4]],1e-3],\
            [1,"svd",True,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [2,"svd",True,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [1,"svd",None,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [2,"svd",None,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [1,"variational",None,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [2,"variational",None,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2])
    def test_ZeroTcorr_MPOscheme3(self,value):
        print "data", value
        J = np.array([[0.0,-0.1,0.0],[-0.1,0.0,-0.3],[0.0,-0.3,0.0]])/constant.au2ev
        nexciton = 0
        procedure = [[1,0],[1,0],[1,0]]
        mol = construct_mol(*value[3])
        
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond = \
        MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton, MPOscheme=2)

        MPSsolver.optimization(iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond,\
                        nexciton, procedure, method="2site")
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        
        if value[2] != None:
            iMPS = [iMPS, iMPSQN, len(iMPS)-1, 0]
            QNargs = [ephtable]
            HMPO = [HMPO, HMPOQN, HMPOQNidx, HMPOQNtot]
        else:
            QNargs = None

        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond,
                "a^\dagger", dipole=True, QNargs=QNargs)
        iMPS = mpslib.MPSdtype_convert(iMPS, QNargs=QNargs)

        nsteps = 50
        dt = 30.0

        autocorr = tMPS.ZeroTCorr(iMPS, HMPO, dipoleMPO, nsteps, dt, ephtable,
                thresh=1.0e-4, cleanexciton=1-nexciton, algorithm=value[0],
                compress_method=value[1], QNargs=QNargs)
        autocorr = np.array(autocorr)

        # scheme3
        iMPS3, iMPSdim3, iMPSQN3, HMPO3, HMPOdim3, HMPOQN3, HMPOQNidx3, HMPOQNtot3, ephtable3, pbond3 = \
        MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton, MPOscheme=3)
        MPSsolver.optimization(iMPS3, iMPSdim3, iMPSQN3, HMPO3, HMPOdim3, ephtable3, pbond3,\
                        nexciton, procedure, method="2site")
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond3[0]):
            HMPO3[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        
        if value[2] != None:
            iMPS3 = [iMPS3, iMPSQN3, len(iMPS3)-1, 0]
            QNargs3 = [ephtable3]
            HMPO3 = [HMPO3, HMPOQN3, HMPOQNidx3, HMPOQNtot3]
        else:
            QNargs3 = None
        
        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond,
                "a^\dagger", dipole=True, QNargs=QNargs3)

        iMPS3 = mpslib.MPSdtype_convert(iMPS3, QNargs=QNargs)

        nsteps = 50
        dt = 30.0

        autocorr3 = tMPS.ZeroTCorr(iMPS3, HMPO3, dipoleMPO, nsteps, dt, ephtable3,\
                thresh=1.0e-4, cleanexciton=1-nexciton, algorithm=value[0],
                compress_method=value[1], QNargs=QNargs3)
        autocorr3 = np.array(autocorr3)

        self.assertTrue(np.allclose(autocorr,autocorr3,rtol=value[4]))


    @data(\
            [1,"svd",True,[[4,4]],1e-3],\
            [2,"svd",True,[[4,4]],1e-3],\
            [1,"svd",None,[[4,4]],1e-3],\
            [2,"svd",None,[[4,4]],1e-3],\
            [1,"variational",None,[[4,4]],1e-3],\
            [2,"variational",None,[[4,4]],1e-3],\
            [1,"svd",True,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [2,"svd",True,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [1,"svd",None,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [2,"svd",None,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [1,"variational",None,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [2,"variational",None,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2])
    def test_FiniteT_spectra_emi(self,value):
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
            QNargs = [ephtable]
            HMPO = [HMPO, HMPOQN, HMPOQNidx, HMPOQNtot]
        else:
            QNargs = None
        
        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond,
                "a", dipole=True, QNargs=QNargs)
        nsteps = 30
        dt = 30.0
        EXMPO, EXMPOdim = tMPS.Max_Entangled_EX_MPO(mol, pbond, norm=True, QNargs=QNargs)
        EXMPO = mpslib.MPSdtype_convert(EXMPO, QNargs=QNargs)

        insteps = 50
        autocorr = tMPS.FiniteT_spectra("emi", mol, pbond, EXMPO, HMPO,
                dipoleMPO, nsteps, dt, ephtable, insteps, thresh=1.0e-3,
                temperature=298, algorithm=value[0], compress_method=value[1],
                QNargs=QNargs)
        
        autocorr = np.array(autocorr)
        
        with open("std_data/tMPS/TTemi_"+str(value[0])+str(value[1])+".npy", 'rb') as f:
            TTemi_std = np.load(f)
        self.assertTrue(np.allclose(autocorr,TTemi_std[0:nsteps],rtol=value[4]))


    @data(\
            [1,"svd",True,[[4,4]],1e-3],\
            [1,"svd",None,[[4,4]],1e-3],\
            [1,"variational",None,[[4,4]],1e-3],\
            [1,"svd",True,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [1,"svd",None,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2],\
            [1,"variational",None,[[4,4],[2,2],[1.e-7,1.e-7]],1e-2])
    def test_FiniteT_spectra_abs(self,value):
        print "data", value
        nexciton = 0
        procedure = [[1,0],[1,0],[1,0]]
        mol = construct_mol(*value[3])
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond = MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        
        if value[2] != None:
            QNargs = [ephtable]
            HMPO = [HMPO, HMPOQN, HMPOQNidx, HMPOQNtot]
        else:
            QNargs = None
        
        
        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond,
                "a^\dagger", dipole=True, QNargs=QNargs)
        GSMPS, GSMPSdim = tMPS.Max_Entangled_GS_MPS(mol, pbond, QNargs=QNargs)
        GSMPO = tMPS.hilbert_to_liouville(GSMPS, QNargs=QNargs)
        GSMPO = mpslib.MPSdtype_convert(GSMPO, QNargs=QNargs)

        nsteps = 50
        dt = 30.0
        autocorr = tMPS.FiniteT_spectra("abs", mol, pbond, GSMPO, HMPO,
                dipoleMPO, nsteps, dt, ephtable, thresh=1.0e-3,
                temperature=298, algorithm=value[0], compress_method=value[1], QNargs=QNargs)
        
        autocorr = np.array(autocorr)

        with open("std_data/tMPS/TTabs_"+str(value[1]+".npy"), 'rb') as f:
            TTabs_std = np.load(f)

        self.assertTrue(np.allclose(autocorr,TTabs_std[0:nsteps],rtol=value[4]))


    @data(["svd",True],["svd",None],["variational",None])
    def test_FiniteT_emi(self,value):
        nexciton = 1
        procedure = [[10,0.4],[20,0.2],[30,0.1],[40,0],[40,0]]
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond = MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        
        if value[1] != None:
            QNargs = [ephtable]
            HMPO = [HMPO, HMPOQN, HMPOQNidx, HMPOQNtot]
        else:
            QNargs = None
        
        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond, "a", dipole=True, QNargs=QNargs)
        nsteps = 30
        dt = 30.0
        EXMPO, EXMPOdim = tMPS.Max_Entangled_EX_MPO(mol, pbond, norm=True, QNargs=QNargs)
        EXMPO = mpslib.MPSdtype_convert(EXMPO, QNargs=QNargs)
    
        insteps = 50
        autocorr = tMPS.FiniteT_emi(mol, pbond, EXMPO, HMPO,
                dipoleMPO, nsteps, dt, ephtable, insteps, thresh=1.0e-3,
                temperature=298, compress_method=value[0], QNargs=QNargs)
        
        autocorr = np.array(autocorr)
        
        with open("std_data/tMPS/TTemi_"+str(1)+str(value[0])+".npy", 'rb') as f:
            TTemi_std = np.load(f)
        self.assertTrue(np.allclose(autocorr,TTemi_std[0:nsteps],rtol=1e-3))


    @data(["svd",True],["svd",None],["variational",None])
    def test_FiniteT_abs(self,value):
        nexciton = 0
        procedure = [[1,0],[1,0],[1,0]]
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond = MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        
        if value[1] != None:
            QNargs = [ephtable]
            HMPO = [HMPO, HMPOQN, HMPOQNidx, HMPOQNtot]
        else:
            QNargs = None
        
        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond,
                "a^\dagger", dipole=True, QNargs=QNargs)
        GSMPS, GSMPSdim = tMPS.Max_Entangled_GS_MPS(mol, pbond, QNargs=QNargs)
        GSMPO = tMPS.hilbert_to_liouville(GSMPS, QNargs=QNargs)
        GSMPO = mpslib.MPSdtype_convert(GSMPO, QNargs=QNargs)

        nsteps = 50
        dt = 30.0
        autocorr = tMPS.FiniteT_abs(mol, pbond, GSMPO, HMPO,
                dipoleMPO, nsteps, dt, ephtable, thresh=1.0e-3,
                temperature=298, compress_method=value[0], QNargs=QNargs)
        
        autocorr = np.array(autocorr)

        with open("std_data/tMPS/TTabs_"+str(value[0]+".npy"), 'rb') as f:
            TTabs_std = np.load(f)

        self.assertTrue(np.allclose(autocorr,TTabs_std[0:nsteps],rtol=1e-3))


if __name__ == "__main__":
    print("Test tMPS")
    unittest.main()
