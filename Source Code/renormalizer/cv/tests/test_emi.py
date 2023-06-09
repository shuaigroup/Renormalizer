# -*- coding: utf-8 -*-
import os

import pytest

from renormalizer.cv import batch_run
from renormalizer.cv.zerot import SpectraZtCV
from renormalizer.cv.finitet import SpectraFtCV
import numpy as np
from renormalizer.tests.parameter import holstein_model, holstein_model4
from renormalizer.cv.tests import cur_dir
from renormalizer.utils import (
    Quantity, CompressConfig, CompressCriteria, EvolveConfig, EvolveMethod)


def test_zt_emi():
    with open(os.path.join(cur_dir, "emi_zt.npy"), "rb") as fin:
        standard_value = np.load(fin)
    # the standard value is plotted over np.arange(-0.11, -0.05, 5.e-5)
    freq_reg = np.arange(-0.11, -0.05, 5.e-5).tolist()
    indx = [520, 529, 661]
    standard_value = standard_value[indx]
    test_freq = [freq_reg[idx] for idx in indx]
    spectra = SpectraZtCV(holstein_model, "emi", 10,
                          5.e-5, rtol=1e-3)
    result = batch_run(test_freq, 1, spectra)
    assert np.allclose(result, standard_value, rtol=1.e-2)


@pytest.mark.parametrize("model", (holstein_model, holstein_model4))
def test_ft_emi(model):
    with open(os.path.join(cur_dir, "emi_ft.npy"), "rb") as fin:
        standard_value = np.load(fin)
    # the standard value is plotted over np.arange(-0.11, -0.05, 5.e-4)
    freq_reg = np.arange(-0.11, -0.05, 5.e-4).tolist()
    test_freq = [freq_reg[52]]
    T = Quantity(298, unit='K')
    standard_value = [standard_value[52]]
    evolve_config = EvolveConfig(method=EvolveMethod.tdvp_ps)
    compress_config = CompressConfig(criteria=CompressCriteria.fixed, max_bonddim=10)
    spectra = SpectraFtCV(model, "emi", 10,
                          5.e-3, T, ievolve_config=evolve_config,
                          icompress_config=compress_config,
                          insteps=10, rtol=1e-3)
    result = batch_run(test_freq, 1, spectra)
    assert np.allclose(result, standard_value, rtol=1.e-2)
