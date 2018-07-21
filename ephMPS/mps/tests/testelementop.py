# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import unittest
import numpy as np
from ephMPS.mps import elementop


class TestElementop(unittest.TestCase):
    def test_phonon_op(self):
        self.assertAlmostEqual(elementop.ph_element_op("b^\dagger b", 3, 3), 3.0)
        self.assertAlmostEqual(elementop.ph_element_op("b^\dagger b", 3, 2), 0.0)
        self.assertAlmostEqual(elementop.ph_element_op("b^\dagger b", 2, 3), 0.0)
        self.assertAlmostEqual(elementop.ph_element_op("b^\dagger b", 0, 0), 0.0)
        
        self.assertAlmostEqual(elementop.ph_element_op("b^\dagger + b", 3, 3), 0.0)
        self.assertAlmostEqual(elementop.ph_element_op("b^\dagger + b", 3, 2), np.sqrt(3.0))
        self.assertAlmostEqual(elementop.ph_element_op("b^\dagger + b", 2, 3), np.sqrt(3.0))
        self.assertAlmostEqual(elementop.ph_element_op("b^\dagger + b", 4, 2), 0.0)
        self.assertAlmostEqual(elementop.ph_element_op("b^\dagger + b", 2, 4), 0.0)
        
        self.assertAlmostEqual(elementop.ph_element_op("Iden", 2, 4), 0.0)
        self.assertAlmostEqual(elementop.ph_element_op("Iden", 4, 2), 0.0)
        self.assertAlmostEqual(elementop.ph_element_op("Iden", 2, 2), 1.0)
        
        with self.assertRaises(AssertionError):
            elementop.ph_element_op("b^\dagger b", 0, -1)
        
    def test_electronic_op(self):
        self.assertAlmostEqual(elementop.e_element_op("a^\dagger", 1, 0), 1.0)
        self.assertAlmostEqual(elementop.e_element_op("a^\dagger", 0, 1), 0.0)
        self.assertAlmostEqual(elementop.e_element_op("a^\dagger", 1, 1), 0.0)
        self.assertAlmostEqual(elementop.e_element_op("a^\dagger", 0, 0), 0.0)

        self.assertAlmostEqual(elementop.e_element_op("a", 1, 0), 0.0)
        self.assertAlmostEqual(elementop.e_element_op("a", 0, 1), 1.0)
        self.assertAlmostEqual(elementop.e_element_op("a", 1, 1), 0.0)
        self.assertAlmostEqual(elementop.e_element_op("a", 0, 0), 0.0)
        
        self.assertAlmostEqual(elementop.e_element_op("a^\dagger a", 1, 0), 0.0)
        self.assertAlmostEqual(elementop.e_element_op("a^\dagger a", 0, 1), 0.0)
        self.assertAlmostEqual(elementop.e_element_op("a^\dagger a", 1, 1), 1.0)
        self.assertAlmostEqual(elementop.e_element_op("a^\dagger a", 0, 0), 0.0)
        
        self.assertAlmostEqual(elementop.e_element_op("Iden", 1, 0), 0.0)
        self.assertAlmostEqual(elementop.e_element_op("Iden", 0, 1), 0.0)
        self.assertAlmostEqual(elementop.e_element_op("Iden", 1, 1), 1.0)
        self.assertAlmostEqual(elementop.e_element_op("Iden", 0, 0), 1.0)
        
        with self.assertRaises(AssertionError):
            elementop.e_element_op("a", 0, 3)

if __name__ == "__main__":
    print("Test elementop")
    unittest.main()
