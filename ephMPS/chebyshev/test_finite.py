# -*- coding: utf-8 -*-
# Author: Tong Jiang <tongjiang1000@gmail.com>

from finitet import SpectraFiniteT
from ephMPS.tests.parameter import mol_list
import numpy as np
from ephMPS.utils import Quantity


freq_reg = np.arange(0.0, 0.15, 5.e-5)
# freq_reg = np.arange(-10, 10, 5.e-5)
insteps = 50
zero_t_greenfunc = SpectraFiniteT(
    mol_list, 'abs', Quantity(298, "K"), insteps, freq_reg, 50, 10)
zero_t_greenfunc.cheb_sum()
