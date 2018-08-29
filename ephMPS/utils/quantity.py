# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li

from __future__ import division

from ephMPS.utils import constant

allowed_units = ['meV', 'eV', 'cm^{-1}', 'a.u.']

au_ratio_dict = {'meV': constant.au2ev * 1e3, 'eV': constant.au2ev, 'cm^{-1}': 1 / constant.cm2au, 'a.u.': 1}


def convert_to_au(num, unit):
    assert unit in allowed_units
    return num / au_ratio_dict[unit]


class Quantity(object):

    @classmethod
    def zero(cls):
        return cls(0, 'a.u.')

    def __init__(self, value, unit):
        super(Quantity, self).__init__()
        self.value = value
        self.unit = unit

    def as_au(self):
        return convert_to_au(self.value, self.unit)

    def __str__(self):
        return '%g %s' % (self.value, self.unit)