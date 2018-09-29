import numpy as np

from ephMPS.model import Phonon, Mol, MolList
from ephMPS.utils import constant, Quantity

elocalex = Quantity(2.67, 'eV')
dipole_abs = 15.45
nmols = 3

# eV
j_matrix = np.array([[0.0, -0.1, -0.2], [-0.1, 0.0, -0.3], [-0.2, -0.3, 0.0]]) / constant.au2ev

omega_quantities = [Quantity(106.51, 'cm^{-1}'), Quantity(1555.55, 'cm^{-1}')]
omega = [[omega_quantities[0], omega_quantities[0]], [omega_quantities[1], omega_quantities[1]]]
displacement_quantities = [Quantity(30.1370, 'a.u.'), Quantity(8.7729, 'a.u.')]
displacement = [[Quantity(0), displacement_quantities[0]], [Quantity(0), displacement_quantities[1]]]
ph_phys_dim = [4, 4]
ph_list = [Phonon(*args) for args in zip(omega, displacement, ph_phys_dim)]

mol_list = MolList([Mol(elocalex, ph_list, dipole_abs)] * nmols)


def custom_mol_list(n_phys_dim, nqboson=None, qbtrunc=None, force3rd=None):
    if nqboson is None:
        nqboson = [1, 1]
    if qbtrunc is None:
        qbtrunc = [0.0, 0.0]
    if force3rd is None:
        force3rd = [None, None]
    ph_list = [Phonon(*args) for args in zip(omega, displacement, n_phys_dim, force3rd, nqboson, qbtrunc)]
    return MolList([Mol(elocalex, ph_list, dipole_abs)] * nmols)
