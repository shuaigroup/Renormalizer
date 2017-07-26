# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>


import numpy as np
import unittest
from ephMPS.configidx import *
from ephMPS.obj import *

'''
an example of no=7, ne=5 and phonon levels [10,10,10,10,1] system
'''
no = 7
ne = 5
ph_dof_list = [10000,1000,100,10,1]

x_std = np.array([[1, 0, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0, 0],
                  [1, 2, 1, 0, 0, 0],
                  [0, 3, 3, 1, 0, 0],
                  [0, 0, 6, 4, 1, 0],
                  [0, 0, 0,10, 5, 1],
                  [0, 0, 0, 0,15, 6],
                  [0, 0, 0, 0, 0,21]],dtype=np.int32)
y_std = np.array([[0,0,0,0,0,0],
                  [1,0,0,0,0,0],
                  [2,1,0,0,0,0],
                  [0,3,1,0,0,0],
                  [0,0,4,1,0,0],
                  [0,0,0,5,1,0],
                  [0,0,0,0,6,0],
                  [0,0,0,0,0,0]],dtype=np.int32)

config_dic = bidict({})
config_dic[1] = (1,0,1,1,0,1,1,2,3,1,4,0)
            

class Test_configidx(unittest.TestCase):
    def test_exciton_string(self):
        x, y = exciton_string(no, ne)
        self.assertTrue(np.array_equal(x,x_std))
        self.assertTrue(np.array_equal(y,y_std))

    def test_exconfig2idx(self):
        self.assertEqual(exconfig2idx([1,1,1,1,1,0,0], y_std), 0)
        self.assertEqual(exconfig2idx([0,0,1,1,1,1,1], y_std), 20)
        self.assertEqual(exconfig2idx([1,0,1,1,0,1,1], y_std), 13)
        self.assertEqual(exconfig2idx([0,1,1,0,1,1,1], y_std), 17)
        with self.assertRaises(AssertionError):
            exconfig2idx([1,1,1,0,1,1,1], y_std)

    def test_idx2exconfig(self):
        self.assertEqual(idx2exconfig(0, x_std),[1,1,1,1,1,0,0])
        self.assertEqual(idx2exconfig(20, x_std),[0,0,1,1,1,1,1])
        self.assertEqual(idx2exconfig(13, x_std),[1,0,1,1,0,1,1])
        self.assertEqual(idx2exconfig(17, x_std),[0,1,1,0,1,1,1])
        with self.assertRaises(AssertionError):
            idx2exconfig(21, x_std)

    def test_idx2phconfig(self):
        self.assertEqual(idx2phconfig(0, ph_dof_list),[0,0,0,0,0])
        self.assertEqual(idx2phconfig(9999, ph_dof_list),[9,9,9,9,0])
        self.assertEqual(idx2phconfig(2314, ph_dof_list),[2,3,1,4,0])
        with self.assertRaises(AssertionError):
            idx2phconfig(10000, ph_dof_list)

    def test_phconfig2idx(self):
        self.assertEqual(phconfig2idx([0,0,0,0,0], ph_dof_list), 0)
        self.assertEqual(phconfig2idx([9,9,9,9,0], ph_dof_list), 9999)
        self.assertEqual(phconfig2idx([2,3,1,4,0], ph_dof_list), 2314)

    def test_config2idx(self):
        self.assertEqual(config2idx([[1,0,1,1,0,1,1],[2,3,1,4,0]],
            indirect=[ph_dof_list, x_std, y_std]), 132314)
        
        self.assertEqual(config2idx([[1,0,1,1,0,1,1],[2,3,1,4,0]],
            direct=[7,config_dic]), 1)
        self.assertEqual(config2idx([[1,0,1,1,0,1,1],[2,3,1,5,0]],
            direct=[7,config_dic]), None)

    def test_idx2config(self):
        self.assertEqual(idx2config(132314, indirect=[ph_dof_list, x_std,
            y_std]),[[1,0,1,1,0,1,1],[2,3,1,4,0]])
        self.assertEqual(idx2config(1, direct=[7,config_dic]),[[1,0,1,1,0,1,1],[2,3,1,4,0]])

if __name__ == "__main__":
    print("Test configidx")
    unittest.main()
