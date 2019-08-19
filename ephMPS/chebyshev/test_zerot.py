# -*- coding: utf-8 -*-
# Author: Tong Jiang <tongjiang1000@gmail.com>

from zeroT import SpectraZeroT
from ephMPS.tests.parameter import mol_list
from ephMPS.utils import OptimizeConfig
import numpy as np


procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]
optimize_config = OptimizeConfig()
optimize_config.procedure = procedure

freq_reg = np.arange(0.05, 0.11, 5.e-5)
zero_t_greenfunc = SpectraZeroT(
    mol_list, 'abs', freq_reg, 30, 10)
zero_t_greenfunc.optimize_config = optimize_config
zero_t_greenfunc.cheb_sum()
