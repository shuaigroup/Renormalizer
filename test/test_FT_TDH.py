# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import copy
import unittest
from ddt import ddt, data
from parameter import *
from ephMPS import TDH
from ephMPS import MPSsolver
from ephMPS.lib import mps as mpslib
from ephMPS import tMPS
from ephMPS import constant


@ddt
class Test_FT_TDH(unittest.TestCase):
    
    def test_FT_DM(self):
        # TDH
        nexciton = 1
        T = 298
        insteps = 100
        DM = TDH.FT_DM(mol, J, nexciton, T, insteps)
        fe, fv = 1, 6
        HAM, Etot, A_el = TDH.construct_H_Ham(mol, J, nexciton, DM, fe, fv, particle="hardcore boson", debug=True)
        self.assertAlmostEqual(Etot, 0.0856330141528)
        occ_std = np.array([[0.20300487], [0.35305247],[0.44394266]])
        self.assertTrue(np.allclose(A_el, occ_std))                
        
        # DMRGresult
        # energy = 0.08534143842580197
        # occ = 0.20881751295568823, 0.35239681740226808, 0.43878566964204374


    @data(\
            [[0.0, 0.0],"emi","std_data/TDH/TDH_FT_emi_0.npy"],\
            [[30.1370, 8.7729],"emi","std_data/TDH/TDH_FT_emi.npy"],\
            [[0.0, 0.0],"abs","std_data/TDH/TDH_FT_abs_0.npy"],\
            [[30.1370, 8.7729],"abs","std_data/TDH/TDH_FT_abs.npy"])
    def test_FT_emi(self,value):
        T = 298.
        insteps = 50
        nsteps = 300
        dt = 30.0
        fe, fv = 1, 6
        
        if value[1] == "emi":
            E_offset = 2.28614053/constant.au2ev
            nexciton = 1
        elif value[1] == "abs":
            E_offset = -2.28614053/constant.au2ev
            nexciton = 0
        
        D_value = np.array(value[0])
        mol = construct_mol(nlevels, D_value=D_value)
        
        DM = TDH.FT_DM(mol, J, nexciton, T, insteps)
        autocorr = TDH.linear_spectra(value[1], mol, J, nexciton, DM, dt, nsteps, fe, fv, E_offset=E_offset, T=T)
        with open(value[2], 'rb') as f:
            std = np.load(f)
        self.assertTrue(np.allclose(autocorr,std))
        

if __name__ == "__main__":
    print("Test FT_TDH")
    unittest.main()
