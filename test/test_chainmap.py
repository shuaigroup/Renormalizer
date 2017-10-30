# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import unittest
from ddt import ddt, data
import ephMPS.chainmap as chainmap
from ephMPS import MPSsolver
from ephMPS.lib import mps as mpslib
from ephMPS import constant
from ephMPS import obj
from ephMPS import tMPS
from parameter import *

@ddt
class Test_chainmap(unittest.TestCase):
    
    @data([1],[3])
    def test_Chain_Map_discrete(self,value):
        
        elocalex = 2.67/constant.au2ev
        dipole_abs = 15.45
        nmols = 3
        # eV
        J = np.array([[0.0,-0.1,-0.2],[-0.1,0.0,-0.3],[-0.2,-0.3,0.0]])/constant.au2ev
        omega_value = np.array([106.51, 1555.55, 1200.0])*constant.cm2au
        D = np.array([30.1370, 8.7729, 20.0])
        nphcoup = np.sqrt(omega_value/2.0)*D
        omega = [{0:omega_value[0], 1:omega_value[0]},{0:omega_value[1], \
            1:omega_value[1]},{0:omega_value[2], 1:omega_value[2]}]
        nexciton=1
        nphs = 3
        
        procedure = [[10,0.4],[20,0.3],[30,0.2],[40,0.1],[40,0]]
        
        nlevels =  [8]*nphs
        nqboson = [value[0]]*nphs
        qbtrunc = [1e-7]*nphs
        phinfo = [list(a) for a in zip(omega, nphcoup, nlevels, nqboson, qbtrunc)]
        mol = []
        for imol in xrange(nmols):
            mol_local = obj.Mol(elocalex, nphs, dipole_abs)
            mol_local.create_ph(phinfo)
            mol.append(mol_local)

        MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
                    MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton, \
                    MPOscheme=2)
        energy1 = MPSsolver.optimization(MPS, MPSdim, MPSQN, MPO, MPOdim, 
                ephtable, pbond, nexciton, procedure, method="2site")
        
        Chain = chainmap.Chain_Map_discrete(mol)
        molnew = chainmap.Chain_Mol(Chain, mol)
        
        MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
                    MPSsolver.construct_MPS_MPO_2(molnew, J, procedure[0][0], nexciton, \
                    MPOscheme=2, rep="chain")

        energy2 = MPSsolver.optimization(MPS, MPSdim, MPSQN, MPO, MPOdim, 
                    ephtable, pbond, nexciton, procedure, method="2site")
        print np.min(energy1), np.min(energy2)
        self.assertAlmostEqual(np.min(energy1), np.min(energy2))


    @data(\
            [[[8,8],[3,3],[1.e-7,1.e-7]],1e-2],\
            [[[8,8],[1,1],[1.e-7,1.e-7]],1e-2],\
            [[[27,27],[3,3],[1.e-7,1.e-7]],1e-2])
    def test_ZeroTcorr(self,value):
        
        nexciton = 0
        procedure = [[20,0.5],[10,0.1],[5,0],[1,0]]
        
        mol = construct_mol(*value[0])
        
        Chain = chainmap.Chain_Map_discrete(mol)
        mol = chainmap.Chain_Mol(Chain, mol)
        
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond = \
        MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton,\
                rep="chain")
        
        MPSsolver.optimization(iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond,\
                        nexciton, procedure, method="2site")
        
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        
        iMPS = [iMPS, iMPSQN, len(iMPS)-1, 0]
        QNargs = [ephtable, False]
        HMPO = [HMPO, HMPOQN, HMPOQNidx, HMPOQNtot]

        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond,
                "a^\dagger", dipole=True, QNargs=QNargs)
        iMPS = mpslib.MPSdtype_convert(iMPS, QNargs=QNargs)

        nsteps = 100
        dt = 10.0

        autocorr = tMPS.ZeroTCorr(iMPS, HMPO, dipoleMPO, nsteps, dt, ephtable,
                thresh=1.0e-3, cleanexciton=1-nexciton, algorithm=2,
                compress_method="svd", QNargs=QNargs)
        autocorr = np.array(autocorr)
        
        with open("std_data/quasiboson/0Tabs_std.npy", 'rb') as f:
            ZeroTabs_std = np.load(f)
        self.assertTrue(np.allclose(autocorr,ZeroTabs_std,rtol=value[1]))

if __name__ == "__main__":
    print("Test chainmap")
    unittest.main()
