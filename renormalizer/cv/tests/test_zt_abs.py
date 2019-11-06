# -*- coding: utf-8 -*-

from renormalizer.cv.zerot import SpectraZtCV
import numpy as np
from renormalizer.tests.parameter import mol_list
import os
from renormalizer.cv.tests import cur_dir
import pytest

def test_zt_abs():

    with open(os.path.join(cur_dir, "standard_value.npy"), "rb") as fin:
        standard_value = np.load(fin)
    # the standard value is plotted over np.arange(0.05, 0.11, 5.e-5)
    freq_reg = np.arange(0.05, 0.11, 5.e-5).tolist()
    indx = [0, 100, 200, 300, 680, 800, 900]
    test_freq = [freq_reg[idx] for idx in indx]
    standard_value = [ivalue[0][0] for ivalue in standard_value[indx]]

    spectra = SpectraZtCV(mol_list, "abs", test_freq, 10,
                          5.e-5)
    # this is all you need to calculate the spectra from frequency domain
    # mol_list used to construct mpo, get ground state,
    # your spectratype, the frequency window, max_bonddim, Lorentzian broadening for spectra
    result = spectra.cv_solve()
    result = [iresult[0][0] for iresult in result]
    assert np.allclose(result, standard_value, rtol=1.e-2)
