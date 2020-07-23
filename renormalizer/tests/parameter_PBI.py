import numpy as np

from renormalizer.utils import Quantity, constant
from renormalizer.model import HolsteinModel, Mol, Phonon


def construct_model(nmols, dmrg_nphs, hartree_nphs) -> HolsteinModel:
    assert dmrg_nphs + hartree_nphs == 10
    elocalex = Quantity(2.13 / constant.au2ev)
    dipole_abs = 1.0

    # cm^-1
    omega_value = (
        np.array(
            [206.0, 211.0, 540.0, 552.0, 751.0, 1325.0, 1371.0, 1469.0, 1570.0, 1628.0]
        )
        * constant.cm2au
    )
    S_value = np.array(
        [0.197, 0.215, 0.019, 0.037, 0.033, 0.010, 0.208, 0.042, 0.083, 0.039]
    )

    # sort from large to small
    gw = np.sqrt(S_value) * omega_value
    idx = np.argsort(gw)[::-1]
    omega_value = omega_value[idx]
    S_value = S_value[idx]

    omega = [[Quantity(x), Quantity(x)] for x in omega_value]
    D_value = np.sqrt(S_value) / np.sqrt(omega_value / 2.0)
    displacement = [[Quantity(0), Quantity(x)] for x in D_value]

    ph_phys_dim = [5] * 10

    # print(dmrg_nphs, hartree_nphs)
    is_hartree = [False] * dmrg_nphs + [True] * hartree_nphs
    ph_list = [
        Phonon(*args[:3], hartree=args[3])
        for args in zip(omega, displacement, ph_phys_dim, is_hartree)
    ]

    model = HolsteinModel([Mol(elocalex, ph_list, dipole_abs)] * nmols, Quantity(500, "cm-1"), )

    return model
