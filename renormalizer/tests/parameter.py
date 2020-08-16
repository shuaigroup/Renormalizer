import numpy as np

from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.utils import constant, Quantity

# todo: make `custom_model or even the whole file a (class) method of `MolList`
elocalex = Quantity(2.67, "eV")
dipole_abs = 15.45
nmols = 3

# eV
_j_matrix = (
    np.array([[0.0, -0.1, -0.2], [-0.1, 0.0, -0.3], [-0.2, -0.3, 0.0]]) / constant.au2ev
)

omega_quantities = [Quantity(106.51, "cm^{-1}"), Quantity(1555.55, "cm^{-1}")]
omega = [
    [omega_quantities[0], omega_quantities[0]],
    [omega_quantities[1], omega_quantities[1]],
]
displacement_quantities = [Quantity(30.1370, "a.u."), Quantity(8.7729, "a.u.")]
displacement = [
    [Quantity(0), displacement_quantities[0]],
    [Quantity(0), displacement_quantities[1]],
]
ph_phys_dim = [4, 4]
ph_list = [Phonon(*args) for args in zip(omega, displacement, ph_phys_dim)]

holstein_model = HolsteinModel([Mol(elocalex, ph_list, dipole_abs)] * nmols, _j_matrix, )
holstein_model4 = holstein_model.switch_scheme(4)

offset = Quantity(2.28614053, "ev") + Quantity(holstein_model.gs_zpe)

def custom_model(
    custom_j_matrix=None,
    n_phys_dim=None,
    dis=None,
    nmols=3,
) -> HolsteinModel:
    if custom_j_matrix is None:
        custom_j_matrix = _j_matrix
    if n_phys_dim is None:
        n_phys_dim = ph_phys_dim
    if dis is None:
        dis = displacement_quantities
    displacement = [[Quantity(0), dis[0]], [Quantity(0), dis[1]]]
    ph_list = [
        Phonon(*args)
        for args in zip(
            omega, displacement, n_phys_dim
        )
    ]
    return HolsteinModel([Mol(elocalex, ph_list, dipole_abs)] * nmols, custom_j_matrix, )
