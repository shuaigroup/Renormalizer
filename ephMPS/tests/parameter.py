import numpy as np

from ephMPS.model import Phonon, Mol, MolList
from ephMPS import constant

elocalex = 2.67 / constant.au2ev
dipole_abs = 15.45
nmols = 3
# eV
j_matrix = np.array([[0.0, -0.1, -0.2], [-0.1, 0.0, -0.3], [-0.2, -0.3, 0.0]]) / constant.au2ev
# cm^-1
omega_value = np.array([106.51, 1555.55]) * constant.cm2au
omega = [[omega_value[0], omega_value[0]], [omega_value[1], omega_value[1]]]
# a.u.
D_value = np.array([30.1370, 8.7729])

D = [[0.0, D_value[0]], [0.0, D_value[1]]]
nphs = 2
ph_phys_dim = [4, 4]

ph_list = [Phonon(*args) for args in zip(omega, D, ph_phys_dim)]

mol_list = MolList([Mol(elocalex, ph_list, dipole_abs)] * nmols)


def custom_mol_list(n_phys_dim, nqboson=None, qbtrunc=None, force3rd=None):
    if nqboson is None:
        nqboson = [1, 1]
    if qbtrunc is None:
        qbtrunc = [0.0, 0.0]
    if force3rd is None:
        force3rd = [None, None]
    ph_list = [Phonon(*args) for args in zip(omega, D, n_phys_dim, force3rd, nqboson, qbtrunc)]
    return MolList([Mol(elocalex, ph_list, dipole_abs)] * nmols)
