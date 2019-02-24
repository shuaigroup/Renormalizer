# -*- coding: utf-8 -*-
from enum import Enum

from ephMPS.utils.rk import RungeKutta

class OptimizeConfig:
    def __init__(self, procedure=None):
        if procedure is None:
            self.procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]
        else:
            self.procedure = procedure
        self.method = "2site"
        self.nroots = 1
        self.inverse = 1.0
        # for dmrg-hartree hybrid
        self.niterations = 20
        self.dmrg_thresh = 1e-5
        self.hartree_thresh = 1e-5


class EvolveMethod(Enum):
    prop_and_compress = "P&C"
    tdvp_ps = "TDVP_PS"
    tdvp_mctdh = "TDVP_MCTDH"
    tdvp_mctdh_new = "TDVP_MCTDHnew"


class EvolveConfig:
    def __init__(self, scheme: EvolveMethod=EvolveMethod.prop_and_compress, rk_config: RungeKutta=None):
        self.scheme = scheme
        # tdvp also requires prop and compress
        if rk_config is None:
            self.rk_config: RungeKutta = RungeKutta()
        else:
            self.rk_config: RungeKutta = rk_config
        self.prop_method = "C_RK4"
        if self.scheme == EvolveMethod.prop_and_compress:
            self.expected_bond_order = None
        else:
            self.expected_bond_order = 50