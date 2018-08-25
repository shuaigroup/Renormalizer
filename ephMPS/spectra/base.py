# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

import numpy as np

from ephMPS.utils import TdMpsJob


class BraKetPairBase(object):

    def __init__(self, bra_mps, ket_mps, *args, **kwargs):
        self.bra_mps = bra_mps
        self.ket_mps = ket_mps
        self.ft = self.calc_ft(*args, **kwargs)

    def calc_ft(self, *args, **kwargs):
        raise NotImplementedError

    def __iter__(self):
        return iter((self.bra_mps, self.ket_mps))


class SpectraTdMpsJobBase(TdMpsJob):

    def __init__(self, i_mps, h_mpo):
        self.i_mps = i_mps
        self.h_mpo = h_mpo
        self.factor = None
        super(SpectraTdMpsJobBase, self).__init__()

    def init_mps(self):
        raise NotImplementedError

    def evolve_single_step(self, evolve_dt):
        raise NotImplementedError

    @property
    def autocorr(self):
        return np.array([pair.ft for pair in self.tdmps_list])

    @property
    def mol_list(self):
        return self.h_mpo.mol_list


