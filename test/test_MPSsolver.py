# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import unittest
from ephMPS import MPSsolver
from ephMPS.lib import mps as mpslib
from ephMPS import constant
from ephMPS import obj
from parameter import *
from ddt import ddt, data

nexciton=1
procedure = [[10,0.4],[20,0.2],[30,0.1],[40,0],[40,0]]

@ddt
class Test_MPSsolver(unittest.TestCase):
    
    def test_construct_MPO(self):
        Mmax = 10
        MPS1, MPSdim1, MPSQN1, MPO1, MPOdim1, MPOQN1, MPOQNidx1, MPOQNtot1, ephtable1, pbond1 = \
            MPSsolver.construct_MPS_MPO_2(mol, J, Mmax, nexciton, MPOscheme=1)
        MPS2, MPSdim2, MPSQN2, MPO2, MPOdim2, MPOQN2, MPOQNidx2, MPOQNtot2, ephtable2, pbond2 = \
            MPSsolver.construct_MPS_MPO_2(mol, J, Mmax, nexciton, MPOscheme=2)
        
        self.assertEqual(ephtable1, ephtable2)
        self.assertEqual(pbond1, pbond2)
        self.assertAlmostEqual( \
            mpslib.dot(MPO1, mpslib.conj(MPO1)) + \
            mpslib.dot(MPO2, mpslib.conj(MPO2)) - \
            mpslib.dot(MPO1, mpslib.conj(MPO2)) - \
            mpslib.dot(MPO2, mpslib.conj(MPO1)), 0.0)
    
        
    def test_construct_MPO_scheme3(self):
        Mmax = 10
        J = np.array([[0.0,-0.1,0.0],[-0.1,0.0,-0.3],[0.0,-0.3,0.0]])/constant.au2ev
        MPS2, MPSdim2, MPSQN2, MPO2, MPOdim2, MPOQN2, MPOQNidx2, MPOQNtot2, ephtable2, pbond2 = \
            MPSsolver.construct_MPS_MPO_2(mol, J, Mmax, nexciton, MPOscheme=2)
        MPS3, MPSdim3, MPSQN3, MPO3, MPOdim3, MPOQN3, MPOQNidx3, MPOQNtot3, ephtable3, pbond3 = \
            MPSsolver.construct_MPS_MPO_2(mol, J, Mmax, nexciton, MPOscheme=3)
        self.assertEqual(ephtable3, ephtable2)
        self.assertEqual(pbond3, pbond2)
        self.assertAlmostEqual( \
            mpslib.dot(MPO3, mpslib.conj(MPO3)) + \
            mpslib.dot(MPO2, mpslib.conj(MPO2)) - \
            mpslib.dot(MPO3, mpslib.conj(MPO2)) - \
            mpslib.dot(MPO2, mpslib.conj(MPO3)), 0.0)


    @data([1],[2])
    def test_optimization(self, value):
        MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
            MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton,
                    MPOscheme=value[0])
        energy = MPSsolver.optimization(MPS, MPSdim, MPSQN, MPO, MPOdim, 
                ephtable, pbond, nexciton, procedure, method="2site")
        self.assertAlmostEqual(np.min(energy)*constant.au2ev, 2.28614053133)

        energy = MPSsolver.optimization(MPS, MPSdim, MPSQN, MPO, MPOdim,
                ephtable, pbond, nexciton, procedure, method="1site")
        self.assertAlmostEqual(np.min(energy)*constant.au2ev, 2.28614053133)

        MPSnew = MPSsolver.clean_MPS("L", MPS, ephtable, nexciton)
        self.assertAlmostEqual( \
            mpslib.dot(MPS, mpslib.conj(MPS)) + \
            mpslib.dot(MPSnew, mpslib.conj(MPSnew)) - \
            mpslib.dot(MPS, mpslib.conj(MPSnew)) - \
            mpslib.dot(MPSnew, mpslib.conj(MPS)), 0.0)

        MPSnew = MPSsolver.clean_MPS("R", MPS, ephtable, nexciton)
        self.assertAlmostEqual( \
            mpslib.dot(MPS, mpslib.conj(MPS)) + \
            mpslib.dot(MPSnew, mpslib.conj(MPSnew)) - \
            mpslib.dot(MPS, mpslib.conj(MPSnew)) - \
            mpslib.dot(MPSnew, mpslib.conj(MPS)), 0.0)


if __name__ == "__main__":
    print("Test MPSsolver")
    unittest.main()
