# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

"""
    some energy units converter
"""

from scipy.constants import physical_constants as c

# 1 a.u. = au2ev eV
au2ev = c["Hartree energy in eV"][0]


# 1 cm^-1 = cm2au a.u.
cm2au = (
    1.0e2
    * c["inverse meter-hertz relationship"][0]
    / c["hartree-hertz relationship"][0]
)
au2cm = 1./ cm2au

# 1 cm^-1 = cm2ev eV
cm2ev = cm2au * au2ev
ev2cm = 1./cm2ev

# 1 fs = fs2au a.u
fs2au = 1.0e-15 / c["atomic unit of time"][0]
K2au = c["kelvin-hartree relationship"][0]
au2K = c["hartree-kelvin relationship"][0]

# atomic mass unit
amu2au = c["atomic mass constant"][0] / c["atomic unit of mass"][0]
angstrom2au = 1e-10 / c["atomic unit of length"][0] 

# nm to au
def nm2au(l):
    return 1.0e7 / l * cm2au


def au2nm(e):
    return 1.0e7 / (e / cm2au)

mobility2au = au2ev * c["atomic unit of time"][0] / (c["atomic unit of length"][0] * 100) ** 2
