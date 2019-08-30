# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import os

import numpy as np
import pytest

from renormalizer.spectra import SpectraFiniteT
from renormalizer.spectra.tests import cur_dir
from renormalizer.tests import parameter
from renormalizer.utils import Quantity


@pytest.mark.parametrize(
    "mol_list, spectratype, std_fname",
    (
        [parameter.mol_list, "abs", "hybrid_FT_abs_pure.npy"],
        [parameter.hybrid_mol_list, "abs", "hybrid_FT_abs_hybrid.npy"],
        [parameter.mol_list, "emi", "hybrid_FT_emi_pure.npy"],
        [parameter.hybrid_mol_list, "emi", "hybrid_FT_emi_hybrid.npy"],
    ),
)
def test_ft_hybrid_dmrg_tdh(mol_list, spectratype, std_fname):
    temperature = Quantity(298, "K")
    insteps = 50
    finite_t = SpectraFiniteT(
        mol_list, spectratype, temperature, insteps, Quantity(2.28614053, "ev")
    )
    finite_t.info_interval = 50
    dt = 30.0
    nsteps = 300
    finite_t.evolve(dt, nsteps)
    with open(os.path.join(cur_dir, std_fname), "rb") as fin:
        std = np.load(fin)
    assert np.allclose(finite_t.autocorr[:nsteps], std[:nsteps], rtol=1e-2)


# from matplotlib import pyplot as plt
# plt.plot(finite_t.autocorr)
# plt.plot(std)
# plt.show()
