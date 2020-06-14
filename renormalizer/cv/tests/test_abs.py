# -*- coding: utf-8 -*-

from renormalizer.cv import batch_run
from renormalizer.cv.zerot import SpectraZtCV
from renormalizer.cv.finitet import SpectraFtCV
import numpy as np
from renormalizer.tests.parameter import mol_list
import os
from renormalizer.cv.tests import cur_dir
from renormalizer.utils import Quantity
import pytest


@pytest.mark.parametrize("method", ("1site", "2site"))
def test_zt_abs(method):
    with open(os.path.join(cur_dir, "abs_zt.npy"), "rb") as fin:
        standard_value = np.load(fin)
    # the standard value is plotted over np.arange(0.05, 0.11, 5.e-5)
    freq_reg = np.arange(0.05, 0.11, 5.e-5).tolist()
    indx = [300, 680, 800, 900]
    test_freq = [freq_reg[idx] for idx in indx]
    standard_value = [ivalue[0][0] for ivalue in standard_value[indx]]
    spectra = SpectraZtCV(mol_list, "abs", 10,
                          5.e-5, method=method, rtol=1e-3)
    result = batch_run(test_freq, 1, spectra)
    assert np.allclose(result, standard_value, rtol=1.e-2)


def test_ft_abs():
    with open(os.path.join(cur_dir, "abs_ft.npy"), "rb") as fin:
        standard_value = np.load(fin)
    # the standard value is plotted over np.arange(0.05, 0.11, 5.e-4)
    freq_reg = np.arange(0.08, 0.10, 2.e-3).tolist()
    indx = [0, 2, 4, 6, 8]
    standard_value = standard_value[indx]
    test_freq = [freq_reg[idx] for idx in indx]
    T = Quantity(298, unit='K')
    spectra = SpectraFtCV(mol_list, "abs",
                          10, 5.e-3, T, rtol=1e-3)
    result = batch_run(test_freq, 1, spectra)
    assert np.allclose(result, standard_value, rtol=1.e-2)
