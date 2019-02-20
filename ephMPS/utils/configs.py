# -*- coding: utf-8 -*-
from enum import Enum


class OptimizeConfig:
    def __init__(self):
        self.procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]
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
    def __init__(self, scheme=EvolveMethod.prop_and_compress):
        self.scheme = scheme
        # tdvp also requires prop and compress
        self.prop_method = "C_RK4"
        if self.scheme == EvolveMethod.prop_and_compress:
            self.expected_bond_order = None
        else:
            self.expected_bond_order = 50

