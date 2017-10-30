# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import unittest
from ephMPS import nparticle
from ephMPS import obj
import itertools

'''
3mol, each mol 1 phonon, nlevels=2
'''
nmols = 3
nphs = 1
elocalex = 1.0
dipole_abs = 1.0
omega_value = np.array([1.0])
omega = [{0:omega_value[0],1:omega_value[0]}]
nphcoup = np.array([1.0])
nlevels =  [2]

mol = []
for imol in xrange(nmols):
    mol_local = obj.Mol(elocalex, nphs, dipole_abs)
    mol_local.create_ph(zip(omega, nphcoup, nlevels))
    mol.append(mol_local)

def construct_config_dict_std(nex,nparticle,config_dict):
    config_dict_std = obj.bidict({})
    idx = -1
    for exconfig in itertools.product([0,1],[0,1],[0,1]):
        if sum(exconfig[0:3]) == nex:
            for phconfig in itertools.product([0,1],[0,1],[0,1]):
                gsvib = 0
                for imol in xrange(3):
                    if exconfig[imol] == 0 and phconfig[imol] == 1:
                        gsvib += 1
                if gsvib <= nparticle-nex:
                    idx += 1
                    config_dict_std[idx] = exconfig + phconfig
    assert len(config_dict_std) == len(config_dict_std.inverse)
    
    assert len(config_dict_std) == len(config_dict)
    for idx in config_dict:
        if idx not in config_dict_std:
            return False

    for iconfig in config_dict.inverse:
        if iconfig not in config_dict_std.inverse:
            return False

    return True


class Test_nparticle(unittest.TestCase):

    def test_construct_config_dict(self):
        
        # 1 exciton, 3-particle
        config_dict = nparticle.construct_config_dict(mol,1,nparticle=False)
        self.assertTrue(construct_config_dict_std(1,3,config_dict))
        config_dict = nparticle.construct_config_dict(mol,1,nparticle=3)
        self.assertTrue(construct_config_dict_std(1,3,config_dict))
                      
        # 1 exciton, 1-particle
        config_dict = nparticle.construct_config_dict(mol,1,nparticle=1)
        self.assertTrue(construct_config_dict_std(1,1,config_dict))

        # 1 exciton, 2-particle
        config_dict = nparticle.construct_config_dict(mol,1,nparticle=2)
        self.assertTrue(construct_config_dict_std(1,2,config_dict))
        
        # 2 exciton, 3-particle
        config_dict = nparticle.construct_config_dict(mol,2,nparticle=False)
        self.assertTrue(construct_config_dict_std(2,3,config_dict))
        config_dict = nparticle.construct_config_dict(mol,2,nparticle=3)
        self.assertTrue(construct_config_dict_std(2,3,config_dict))
        
        # 2 exciton, 2-particle
        config_dict = nparticle.construct_config_dict(mol,2,nparticle=2)
        self.assertTrue(construct_config_dict_std(2,2,config_dict))

        # 3 exciton, 3-particle
        config_dict = nparticle.construct_config_dict(mol,3,nparticle=False)
        self.assertTrue(construct_config_dict_std(3,3,config_dict))
        config_dict = nparticle.construct_config_dict(mol,3,nparticle=3)
        self.assertTrue(construct_config_dict_std(3,3,config_dict))

if __name__ == "__main__":
    print("Test nparticle")
    unittest.main()
