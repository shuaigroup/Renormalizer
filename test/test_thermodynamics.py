# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import unittest
from ddt import ddt, data
from parameter import *
from ephMPS import tMPS
from ephMPS import RK
from ephMPS import MPSsolver
from ephMPS.lib import mps as mpslib


@ddt
class Test_thermodynamics(unittest.TestCase):
    
    def test_it(self):
        
        nexciton = 1
        procedure = [[1,0]]
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond = \
            MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  2.28614053/constant.au2ev
        
        QNargs = [ephtable, False]
        HMPO = [HMPO, HMPOQN, HMPOQNidx, HMPOQNtot]
        
        EXMPO, EXMPOdim = tMPS.Max_Entangled_EX_MPO(mol, pbond, norm=True, QNargs=QNargs)
        EXMPO = mpslib.MPSdtype_convert(EXMPO, QNargs=QNargs)

        insteps = 50
        rk = RK.Runge_Kutta(method="C_RK4", adaptive=False, rtol=1e-5)

        ketMPO = tMPS.thermal_prop(rk, EXMPO, HMPO, insteps, ephtable,\
                thresh=1e-3, temperature=298, compress_method="svd", \
                QNargs=QNargs, normalize=1.0)
        
        np.save("1e-3", ketMPO)

if __name__ == "__main__":
    print("Test thermodynamics")
    unittest.main()

