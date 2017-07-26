# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

'''
    some energy units converter
'''

import scipy.constants 

# 1 a.u. = au2ev eV
au2ev = scipy.constants.physical_constants["Hartree energy in eV"][0]

# 1 cm^-1 = cm2au a.u.
cm2au = 1.0E2 * \
scipy.constants.physical_constants["inverse meter-hertz relationship"][0] / \
scipy.constants.physical_constants["hartree-hertz relationship"][0]

# 1 cm^-1 = cm2ev eV
cm2ev = cm2au * au2ev

# kelvin to beta  au^-1 
def T2beta(temperature):
    '''
    temperature to beta
    '''
    beta = 1.0 / temperature / \
    scipy.constants.physical_constants["kelvin-hartree relationship"][0]
    
    return beta
