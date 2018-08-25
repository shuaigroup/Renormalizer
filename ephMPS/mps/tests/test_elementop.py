# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import unittest

from ddt import ddt, data, unpack
import numpy as np

from ephMPS.mps import elementop


@ddt
class TestElementop(unittest.TestCase):

    @data(
        ['b^\dagger b', 3, 3, 3.0],
        ['b^\dagger b', 3, 2, 0.0],
        ['b^\dagger b', 2, 3, 0.0],
        ['b^\dagger b', 0, 0, 0.0],

        ['b^\dagger + b', 3, 3, 0.0],
        ['b^\dagger + b', 3, 2, np.sqrt(3.0)],
        ['b^\dagger + b', 2, 3, np.sqrt(3.0)],
        ['b^\dagger + b', 4, 2, 0.0],
        ['b^\dagger + b', 2, 4, 0.0],

        ['Iden', 2, 4, 0.0],
        ['Iden', 4, 2, 0.0],
        ['Iden', 2, 2, 1.0],
    )
    @unpack
    def test_phonon_op(self, op, bra, ket, value):
        self.assertAlmostEqual(elementop.ph_element_op(op, bra, ket), value)

    def test_phonon_exception(self):
        with self.assertRaises(AssertionError):
            elementop.ph_element_op("b^\dagger b", 0, -1)

    @data(
        ['a^\dagger', 1, 0, 1.0],
        ['a^\dagger', 0, 1, 0.0],
        ['a^\dagger', 1, 1, 0.0],
        ['a^\dagger', 0, 0, 0.0],

        ['a', 1, 0, 0.0],
        ['a', 0, 1, 1.0],
        ['a', 1, 1, 0.0],
        ['a', 0, 0, 0.0],

        ['a^\dagger a', 1, 0, 0.0],
        ['a^\dagger a', 0, 1, 0.0],
        ['a^\dagger a', 1, 1, 1.0],
        ['a^\dagger a', 0, 0, 0.0],

        ['Iden', 1, 0, 0.0],
        ['Iden', 0, 1, 0.0],
        ['Iden', 1, 1, 1.0],
        ['Iden', 0, 0, 1.0],
    )
    @unpack
    def test_electronic_op(self, op, bra, ket, value):
        self.assertAlmostEqual(elementop.e_element_op(op, bra, ket), value)

    def test_electronic_exception(self):
        with self.assertRaises(AssertionError):
            elementop.e_element_op("a", 0, 3)


if __name__ == "__main__":
    print("Test elementop")
    unittest.main()
