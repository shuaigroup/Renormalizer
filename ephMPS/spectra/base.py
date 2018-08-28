# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

import numpy as np

from ephMPS.utils import TdMpsJob


class BraKetPair(object):

    def __init__(self, bra_mps, ket_mps, *args, **kwargs):
        self.bra_mps = bra_mps
        self.ket_mps = ket_mps
        self.ft = self.calc_ft(*args, **kwargs)

    def calc_ft(self, *args, **kwargs):
        return self.bra_mps.conj().dot(self.ket_mps)

    def __str__(self):
        if np.iscomplex(self.ft):
            # if negative, sign is included in the imag part
            sign = '+' if 0 <= self.ft.imag else ''
            ft_str = '%g%s%gj' % (self.ft.real, sign, self.ft.imag)
        else:
            ft_str = '%g' % self.ft
        return 'bra: %s, ket: %s, ft: %s' % (self.bra_mps, self.ket_mps, ft_str)

    def __iter__(self):
        return iter((self.bra_mps, self.ket_mps))


class SpectraTdMpsJobBase(TdMpsJob):

    def __init__(self, i_mps, h_mpo):
        self.i_mps = i_mps
        self.h_mpo = h_mpo
        self.norm = None
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

