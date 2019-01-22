# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

import numpy as np

from ephMPS.mps import solver, Mps, Mpo
from ephMPS.utils import TdMpsJob, Quantity


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

    def __init__(self, mol_list, spectratype, temperature, scheme=2, offset=Quantity(0)):
        self.mol_list = mol_list
        assert spectratype in ["emi", "abs"]
        self.spectratype = spectratype
        if spectratype == "emi":
            self.nexciton = 1
        else:
            self.nexciton = 0
        self.temperature = temperature
        self.h_mpo = Mpo(mol_list, scheme=scheme, offset=offset)
        super(SpectraTdMpsJobBase, self).__init__()

    @property
    def autocorr(self):
        return np.array([pair.ft for pair in self.tdmps_list])

