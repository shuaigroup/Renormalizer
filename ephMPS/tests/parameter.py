import numpy as np

from ephMPS.model import Phonon, Mol
from ephMPS import constant


elocalex = 2.67/constant.au2ev
dipole_abs = 15.45
nmols = 3
# eV
J = np.array([[0.0,-0.1,-0.2],[-0.1,0.0,-0.3],[-0.2,-0.3,0.0]])/constant.au2ev
# cm^-1
omega_value = np.array([106.51, 1555.55])*constant.cm2au
omega = [{0:omega_value[0],1:omega_value[0]},{0:omega_value[1],1:omega_value[1]}]
# a.u.
D_value = np.array([30.1370, 8.7729])

D = [{0:0.0,1:D_value[0]},{0:0.0,1:D_value[1]}]
nphs = 2
nlevels = [4,4]

ph_list = [Phonon(*args) for args in zip(omega, D, nlevels)]

mol = [Mol(elocalex, dipole_abs, ph_list)] * nmols

