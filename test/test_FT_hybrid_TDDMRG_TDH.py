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
class Test_FT_hybrid_TDDMRG_TDH(unittest.TestCase):
    
    @data(\
            [mol_hybrid,0.0853441664951,[0.20881609,0.35239430,0.43878960]],\
            [mol_pure,0.0853413581416,[0.20881782,0.35239674,0.43878545]])
    def test_DM_hybrid_TDDMRG_TDH(self,value):

        TDH.construct_Ham_vib(value[0], hybrid=True)
        T = 298.
        nexciton = 1
        nsteps = 100
        
        MPS, MPSdim, MPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond = \
            MPSsolver.construct_MPS_MPO_2(value[0], J, dmrg_procedure[0][0], nexciton)
        
        HMPO_init = [HMPO, HMPOQN, HMPOQNidx, HMPOQNtot]
        QNargs = [ephtable, False]
        #QNargs = None
        
        rk = RK.Runge_Kutta("C_RK4")

        MPS, DMH = hybrid_TDDMRG_TDH.FT_DM_hybrid_TDDMRG_TDH(rk, value[0], J,\
                HMPO_init, nexciton, T, \
                nsteps, pbond, ephtable, thresh=1e-3, cleanexciton=1, QNargs=QNargs)
        
        if QNargs is not None:   
            MPS = MPS[0]

        MPO, HAM, Etot, A_el = \
            hybrid_TDDMRG_TDH.construct_hybrid_Ham(value[0], J, HMPO_init, MPS, \
                    DMH, debug=True, QNargs=QNargs)
        
        self.assertAlmostEqual(Etot, value[1])
        occ_std = np.array(value[2])
        self.assertTrue(np.allclose(A_el, occ_std))                
        

    @data(\
            [mol_pure,"abs","std_data/hybrid_TDDMRG_TDH/hybrid_FT_abs_pure.npy"],\
            [mol_hybrid,"abs","std_data/hybrid_TDDMRG_TDH/hybrid_FT_abs_hybrid.npy"],\
            [mol_pure,"emi","std_data/hybrid_TDDMRG_TDH/hybrid_FT_emi_pure.npy"],\
            [mol_hybrid,"emi","std_data/hybrid_TDDMRG_TDH/hybrid_FT_emi_hybrid.npy"])
    def test_FiniteT_spectra_TDDMRG_TDH(self,value):
        
        TDH.construct_Ham_vib(value[0], hybrid=True)
        T = 298.
        insteps = 50
        dt = 30.
        nsteps = 300
        if value[1] == "abs":
            E_offset = -2.28614053/constant.au2ev
            nexciton = 0
        else:
            E_offset = 2.28614053/constant.au2ev
            nexciton = 1

        MPS, MPSdim, MPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond = \
            MPSsolver.construct_MPS_MPO_2(value[0], J, dmrg_procedure[0][0], nexciton)
        
        HMPO_init = [HMPO, HMPOQN, HMPOQNidx, HMPOQNtot]
        QNargs = [ephtable, False]
        #QNargs = None

        rk = RK.Runge_Kutta("C_RK4")
        #rk = RK.Runge_Kutta(method="RKF45", rtol=1e-3, adaptive=True)
        setup = tMPS.prop_setup(rk)

        autocorr = hybrid_TDDMRG_TDH.FiniteT_spectra_TDDMRG_TDH(setup, value[1], \
                T, value[0], J, HMPO_init, nsteps, \
                dt, insteps, pbond, ephtable, thresh=1e-3, ithresh=1e-3, E_offset=E_offset, QNargs=QNargs)
        
        with open(value[2], 'rb') as f:
            std = np.load(f)
        self.assertTrue(np.allclose(autocorr,std))

if __name__ == "__main__":
    print("Test FT_hybrid_TDDMRG_TDH")
    unittest.main()
