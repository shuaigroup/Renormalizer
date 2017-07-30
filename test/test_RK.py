# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import unittest
from ephMPS import RK

class Test_RK(unittest.TestCase):

    def test_RK(self):
        tableau = RK.runge_kutta_explicit_tableau("Forward_Euler")
        vout = RK.runge_kutta_explicit_coefficient(tableau)
        self.assertTrue(np.allclose(vout,[1.0,1.0]))

        tableau = RK.runge_kutta_explicit_tableau("Heun_RK2")
        vout = RK.runge_kutta_explicit_coefficient(tableau)
        self.assertTrue(np.allclose(vout,[1.0,1.0,0.5]))

        tableau = RK.runge_kutta_explicit_tableau("Ralston_RK2")
        vout = RK.runge_kutta_explicit_coefficient(tableau)
        self.assertTrue(np.allclose(vout,[1.0,1.0,0.5]))
        
        tableau = RK.runge_kutta_explicit_tableau("midpoint_RK2")
        vout = RK.runge_kutta_explicit_coefficient(tableau)
        self.assertTrue(np.allclose(vout,[1.0,1.0,0.5]))
        
        tableau = RK.runge_kutta_explicit_tableau("Kutta_RK3")
        vout = RK.runge_kutta_explicit_coefficient(tableau)
        self.assertTrue(np.allclose(vout,[1.,1.,0.5,0.16666667]))

        tableau = RK.runge_kutta_explicit_tableau("C_RK4")
        vout = RK.runge_kutta_explicit_coefficient(tableau)
        self.assertTrue(np.allclose(vout,[1.,1.,0.5,0.16666667,0.04166667]))
        
        tableau = RK.runge_kutta_explicit_tableau("38rule_RK4")
        vout = RK.runge_kutta_explicit_coefficient(tableau)
        self.assertTrue(np.allclose(vout,[1.,1.,0.5,0.16666667,0.04166667]))

if __name__ == "__main__":
    print("Test RK")
    unittest.main()
