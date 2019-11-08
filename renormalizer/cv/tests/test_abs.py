# -*- coding: utf-8 -*-

from renormalizer.cv.zerot import SpectraZtCV
from renormalizer.cv.finitet import SpectraFtCV
import numpy as np
from renormalizer.tests.parameter import mol_list
import os
from renormalizer.cv.tests import cur_dir
from renormalizer.utils import Quantity, CompressConfig
import pytest


def test_zt_abs(switch_to_64backend):

    with open(os.path.join(cur_dir, "zt_standard.npy"), "rb") as fin:
        standard_value = np.load(fin)
    # the standard value is plotted over np.arange(0.05, 0.11, 5.e-5)
    freq_reg = np.arange(0.05, 0.11, 5.e-5).tolist()
    indx = [0, 100, 200, 300, 680, 800, 900]
    test_freq = [freq_reg[idx] for idx in indx]
    standard_value = [ivalue[0][0] for ivalue in standard_value[indx]]

    spectra = SpectraZtCV(mol_list, "abs", test_freq, 10,
                          5.e-5, cores=4)
    spectra.init_oper()
    spectra.init_mps()
    result = spectra.run()
    result = [iresult[0][0] for iresult in result]
    assert np.allclose(result, standard_value, rtol=1.e-2)


def test_ft_abs(switch_to_64backend):
    with open(os.path.join(cur_dir, "ft_standard.npy"), "rb") as fin:
        standard_value = np.load(fin)
    # the standard value is plotted over np.arange(0.05, 0.11, 5.e-4)
    freq_reg = np.arange(0.08, 0.10, 5.e-4).tolist()
    indx = [0, 7, 8, 22]
    standard_value = standard_value[indx]
    test_freq = [freq_reg[idx] for idx in indx]
    T = Quantity(298, unit='K')
    spectra = SpectraFtCV(mol_list, "abs", T, test_freq,
                          10, 1.e-3, cores=1)
    spectra.init_oper()
    spectra.init_mps()
    result = spectra.run()
    result = [iresult[0] for iresult in result]
    assert np.allclose(result, standard_value, rtol=1.e-2)
