# -*- coding: utf-8 -*-

import numpy as np

from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.utils import Quantity
from renormalizer.transport import EDGE_THRESHOLD

mol_num = 13
ph_list = [
    Phonon.simple_phonon(Quantity(omega, "cm^{-1}"), Quantity(displacement, "a.u."), 4)
    for omega, displacement in [[1e-10, 1e-10]]
]
j_constant = Quantity(0.8, "eV")
band_limit_model = HolsteinModel([Mol(Quantity(0), ph_list)] * mol_num, j_constant, 3)

# the temperature should be compatible with the low vibration frequency in TestBandLimitFiniteT
# otherwise underflow happens in exact propagator
low_t = Quantity(1e-7, "K")


def get_analytical_r_square(time_series: np.ndarray):
    return 2 * (j_constant.as_au()) ** 2 * time_series ** 2


def assert_band_limit(ct, rtol):
    analytical_r_square = get_analytical_r_square(ct.evolve_times_array)
    # has evolved to the edge but not too large
    assert EDGE_THRESHOLD < ct.latest_mps.e_occupations[0] < 0.1
    # value OK
    assert np.allclose(analytical_r_square, ct.r_square_array, rtol=rtol)