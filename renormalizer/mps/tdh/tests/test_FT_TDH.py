# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
import os

import numpy as np
import pytest

from renormalizer.mps.tdh import tdh
from renormalizer.tests.parameter import hartree_holstein_model, custom_model
from renormalizer.mps.tdh.tests import cur_dir
from renormalizer.utils import Quantity, constant


def test_FT_DM():
    # TDH
    nexciton = 1
    T = Quantity(298, "K")
    insteps = 100
    tdHartree = tdh.Dynamics(hartree_holstein_model, temperature=T, insteps=insteps)
    DM = tdHartree._FT_DM(nexciton)
    HAM, Etot, A_el = tdHartree.construct_H_Ham(nexciton, DM, debug=True)
    assert Etot == pytest.approx(0.0856330141528)
    occ_std = np.array([[0.20300487], [0.35305247], [0.44394266]])
    assert np.allclose(A_el, occ_std)

    # DMRGresult
    # energy = 0.08534143842580197
    # occ = 0.20881751295568823, 0.35239681740226808, 0.43878566964204374


@pytest.mark.parametrize(
    "D_value, spectratype, std_path",
    (
        [[0.0, 0.0], "emi", "TDH_FT_emi_0.npy"],
        [[30.1370, 8.7729], "emi", "TDH_FT_emi.npy"],
        [[0.0, 0.0], "abs", "TDH_FT_abs_0.npy"],
        [[30.1370, 8.7729], "abs", "TDH_FT_abs.npy"],
    ),
)
def test_FT_spectra(D_value, spectratype, std_path):

    if spectratype == "emi":
        E_offset = 2.28614053 / constant.au2ev
    elif spectratype == "abs":
        E_offset = -2.28614053 / constant.au2ev
    else:
        assert False

    model = custom_model(
        None, dis=[Quantity(d) for d in D_value], hartrees=[True, True]
    )

    T = Quantity(298, "K")
    insteps = 50
    spectra = tdh.LinearSpectra(
        spectratype, model, E_offset=E_offset, temperature=T, insteps=insteps
    )
    nsteps = 300 - 1
    dt = 30.0
    spectra.evolve(dt, nsteps)
    with open(os.path.join(cur_dir, std_path), "rb") as f:
        std = np.load(f)
    assert np.allclose(spectra.autocorr, std)
