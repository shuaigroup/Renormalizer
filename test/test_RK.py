# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import unittest
from ephMPS import RK

class Test_RK(unittest.TestCase):

    def test_RK(self):
        rk = RK.Runge_Kutta(method="Forward_Euler")
        vout = rk.runge_kutta_explicit_coefficient()
        self.assertTrue(np.allclose(vout,[1.0,1.0]))
        vout = rk.Te_coeff()
        self.assertTrue(np.allclose(vout,[1.0,1.0]))

        rk = RK.Runge_Kutta(method="Heun_RK2")
        vout = rk.runge_kutta_explicit_coefficient()
        self.assertTrue(np.allclose(vout,[1.0,1.0,0.5]))
        vout = rk.Te_coeff()
        self.assertTrue(np.allclose(vout,[1.0,1.0,0.5]))

        rk = RK.Runge_Kutta(method="Ralston_RK2")
        vout = rk.runge_kutta_explicit_coefficient()
        self.assertTrue(np.allclose(vout,[1.0,1.0,0.5]))
        vout = rk.Te_coeff()
        self.assertTrue(np.allclose(vout,[1.0,1.0,0.5]))
        
        rk = RK.Runge_Kutta(method="midpoint_RK2")
        vout = rk.runge_kutta_explicit_coefficient()
        self.assertTrue(np.allclose(vout,[1.0,1.0,0.5]))
        vout = rk.Te_coeff()
        self.assertTrue(np.allclose(vout,[1.0,1.0,0.5]))
        
        rk = RK.Runge_Kutta(method="Kutta_RK3")
        vout = rk.runge_kutta_explicit_coefficient()
        self.assertTrue(np.allclose(vout,[1.,1.,0.5,0.16666667]))
        vout = rk.Te_coeff()
        self.assertTrue(np.allclose(vout,[1.,1.,0.5,0.16666667]))

        rk = RK.Runge_Kutta(method="C_RK4")
        vout = rk.runge_kutta_explicit_coefficient()
        self.assertTrue(np.allclose(vout,[1.,1.,0.5,0.16666667,0.04166667]))
        vout = rk.Te_coeff()
        self.assertTrue(np.allclose(vout,[1.,1.,0.5,0.16666667,0.04166667]))
        
        rk = RK.Runge_Kutta(method="38rule_RK4")
        vout = rk.runge_kutta_explicit_coefficient()
        self.assertTrue(np.allclose(vout,[1.,1.,0.5,0.16666667,0.04166667]))
        vout = rk.Te_coeff()
        self.assertTrue(np.allclose(vout,[1.,1.,0.5,0.16666667,0.04166667]))
        
        #rk = RK.Runge_Kutta("Fehlberg5")
        #vout = RK.runge_kutta_explicit_coefficient(rk)
        #
        #rk = RK.Runge_Kutta("RKF45")
        #vout = RK.runge_kutta_explicit_coefficient(rk)

if __name__ == "__main__":
    print("Test RK")
    unittest.main()
