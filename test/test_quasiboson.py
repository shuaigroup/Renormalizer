# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import unittest
from ephMPS import tMPS
from ephMPS import MPSsolver
from ephMPS.lib import mps as mpslib
from ephMPS import constant
from ephMPS import obj
from parameter import *
from ddt import ddt, data

nexciton=1
procedure = [[10,0.4],[20,0.2],[30,0.1],[40,0],[40,0]]


@ddt
class Test_quasiboson(unittest.TestCase):

    def test_quasiboson_constructMPO(self):
        
        mol = construct_mol([4,4])
        MPS1, MPSdim1, MPSQN1, MPO1, MPOdim1, MPOQN1, MPOQNidx1, MPOQNtot1, ephtable1, pbond1 = \
            MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton, MPOscheme=2)
        
        mol = construct_mol([4,4], [2,2], [1e-7, 1e-7])
        MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
            MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton,
                    MPOscheme=2)
        
        # merge the decomposed MPO  
        MPOmerge = []
        impo = 0
        for imol in xrange(nmols):
            MPOmerge.append(MPO[impo])
            impo += 1
            for iph in xrange(mol[imol].nphs):
                MPOmerge.append(np.einsum("abcd, defg -> abecfg", MPO[impo],MPO[impo+1]). reshape(MPO[impo].shape[0],
                    4, 4, MPO[impo+1].shape[-1]))
                impo += 2

        self.assertAlmostEqual( \
            mpslib.distance(MPO1, MPOmerge), 0.0)

        energy = MPSsolver.optimization(MPS, MPSdim, MPSQN, MPO, MPOdim, 
                ephtable, pbond, nexciton, procedure, method="2site")
        self.assertAlmostEqual(np.min(energy)*constant.au2ev, 2.28614053133)

        energy = MPSsolver.optimization(MPS, MPSdim, MPSQN, MPO, MPOdim,
                ephtable, pbond, nexciton, procedure, method="1site")
        self.assertAlmostEqual(np.min(energy)*constant.au2ev, 2.28614053133)

        MPSnew = MPSsolver.clean_MPS("L", MPS, ephtable, nexciton)
        self.assertAlmostEqual( \
            mpslib.distance(MPSnew, mpslib.conj(MPS)), 0.0) 

        MPSnew = MPSsolver.clean_MPS("R", MPS, ephtable, nexciton)
        self.assertAlmostEqual( \
            mpslib.distance(MPSnew, mpslib.conj(MPS)), 0.0) 
    
    
    @data(\
        [[[64,64]], [[64,64], [6,6], [1e-7, 1e-7]], [[64,64], [6,1], [1e-7,1e-7]]] ,\
        [[[27,27]], [[27,27], [3,3], [1e-7, 1e-7]], [[27,27], [3,1], [1e-7,1e-7]]])
    def test_quasiboson_MPSsolver(self, value):
        # normal boson
        mol = construct_mol(*value[0])
        MPS1, MPSdim1, MPSQN1, MPO1, MPOdim1, MPOQN1, MPOQNidx1, MPOQNtot1, ephtable1, pbond1 = \
            MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton, MPOscheme=2)

        # quasiboson
        mol = construct_mol(*value[1])
        MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
            MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton, \
                    MPOscheme=2)
        
        # quasiboson + normal boson
        mol = construct_mol(*value[2]) 
        MPS2, MPSdim2, MPSQN2, MPO2, MPOdim2, MPOQN2, MPOQNidx2, MPOQNtot2,\
        ephtable2, pbond2 = MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton, MPOscheme=2)
        
        energy1 = MPSsolver.optimization(MPS1, MPSdim1, MPSQN1, MPO1,
                MPOdim1, ephtable1, pbond1, nexciton, procedure, method="1site")
        energy = MPSsolver.optimization(MPS, MPSdim, MPSQN, MPO, MPOdim,
                ephtable, pbond, nexciton, procedure, method="1site")
        energy2 = MPSsolver.optimization(MPS2, MPSdim2, MPSQN2, MPO2,
                MPOdim2, ephtable2, pbond2, nexciton, procedure, method="1site")
        self.assertAlmostEqual(np.min(energy), np.min(energy1))
        self.assertAlmostEqual(np.min(energy2), np.min(energy1))

        energy1 = MPSsolver.optimization(MPS1, MPSdim1, MPSQN1, MPO1,
                MPOdim1, ephtable1, pbond1, nexciton, procedure, method="2site")
        energy = MPSsolver.optimization(MPS, MPSdim, MPSQN, MPO, MPOdim,
                ephtable, pbond, nexciton, procedure, method="2site")
        energy2 = MPSsolver.optimization(MPS2, MPSdim2, MPSQN2, MPO2,
                MPOdim2, ephtable2, pbond2, nexciton, procedure, method="2site")
        self.assertAlmostEqual(np.min(energy), np.min(energy1))
        self.assertAlmostEqual(np.min(energy2), np.min(energy1))
        

    @data(\
            [2,"svd",True,[[8,8],[3,3],[1.e-7,1.e-7]],1e-2],\
            [2,"svd",None,[[8,8],[3,3],[1.e-7,1.e-7]],1e-2],\
            [2,"svd",True,[[9,9],[2,2],[1.e-7,1.e-7]],1e-2],\
            [2,"svd",None,[[9,9],[2,2],[1.e-7,1.e-7]],1e-2])
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
            QNargs = [ephtable, False]
            HMPO = [HMPO, HMPOQN, HMPOQNidx, HMPOQNtot]
        else:
            QNargs = None
        
        dipoleMPO, dipoleMPOdim = MPSsolver.construct_onsiteMPO(mol, pbond,
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
        
        with open("std_data/quasiboson/TTemi_std.npy", 'rb') as f:
            TTemi_std = np.load(f)
        self.assertTrue(np.allclose(autocorr,TTemi_std[0:nsteps],rtol=value[4]))


    @data(\
            [2,"svd",True,[[8,8],[3,3],[1.e-7,1.e-7]],1e-2],\
            [2,"svd",None,[[8,8],[3,3],[1.e-7,1.e-7]],1e-2],\
            [2,"svd",True,[[27,27],[3,3],[1.e-7,1.e-7]],1e-2],\
            [2,"svd",None,[[27,27],[3,3],[1.e-7,1.e-7]],1e-2])
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
            QNargs = [ephtable, False]
            HMPO = [HMPO, HMPOQN, HMPOQNidx, HMPOQNtot]
        else:
            QNargs = None

        dipoleMPO, dipoleMPOdim = MPSsolver.construct_onsiteMPO(mol, pbond,
                "a^\dagger", dipole=True, QNargs=QNargs)
        iMPS = mpslib.MPSdtype_convert(iMPS, QNargs=QNargs)
        

        nsteps = 100
        dt = 10.0

        autocorr = tMPS.ZeroTCorr(iMPS, HMPO, dipoleMPO, nsteps, dt, ephtable,
                thresh=1.0e-3, cleanexciton=1-nexciton, algorithm=value[0],
                compress_method=value[1], QNargs=QNargs)
        autocorr = np.array(autocorr)

        with open("std_data/quasiboson/0Tabs_std.npy", 'rb') as f:
            ZeroTabs_std = np.load(f)
        self.assertTrue(np.allclose(autocorr,ZeroTabs_std,rtol=value[4]))

if __name__ == "__main__":
    print("Test quasiboson")
    unittest.main()
