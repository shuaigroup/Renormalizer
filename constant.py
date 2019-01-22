# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

'''
    some energy units converter
'''

import scipy.constants 

# 1 a.u. = au2ev eV
au2ev = scipy.constants.physical_constants["Hartree energy in eV"][0]
ev2au = 1./au2ev

# 1 cm^-1 = cm2au a.u.
cm2au = 1.0E2 * \
scipy.constants.physical_constants["inverse meter-hertz relationship"][0] / \
scipy.constants.physical_constants["hartree-hertz relationship"][0]
au2cm = 1./cm2au

# 1 cm^-1 = cm2ev eV
cm2ev = cm2au * au2ev
ev2cm = 1./cm2ev

# 1 fs = fs2au a.u
fs2au =  1.0e-15 / scipy.constants.physical_constants["atomic unit of time"][0]
au2fs = 1./fs2au

K2au = scipy.constants.physical_constants["kelvin-hartree relationship"][0]
au2K = scipy.constants.physical_constants["hartree-kelvin relationship"][0]


# kelvin to beta  au^-1 
def T2beta(temperature):
    '''
    temperature to beta
    '''
    beta = 1.0 / temperature / \
    scipy.constants.physical_constants["kelvin-hartree relationship"][0]
    
    return beta

def beta2T(beta):
    '''
    beta to temperature
    '''
    T = 1. / beta / \
    scipy.constants.physical_constants["kelvin-hartree relationship"][0]
    
    return T

# nm to au
def nm2au(l):
    return 1.e7/l*cm2au

def au2nm(e):
    return 1.e7/(e/cm2au)

