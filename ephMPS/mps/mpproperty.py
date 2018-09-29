# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

from cached_property import cached_property
import numpy as np

from ephMPS.mps.mp import MatrixProduct
from ephMPS.mps.mpobase import MpoBase


def invalidate_cache_decorator(f):
    def wrapper(self, *args, **kwargs):
        ret = f(self, *args, **kwargs)
        assert isinstance(ret, self.__class__)
        ret.invalidate_cache()
        return ret
    return wrapper


class MpProperty(MatrixProduct):

    def calc_e_occupation(self, idx):
        return self.expectation(MpoBase.onsite(self.mol_list, 'a^\dagger a', mol_idx_set={idx}))

    def calc_ph_occupation(self, mol_idx, ph_idx):
        return self.expectation(MpoBase.ph_occupation_mpo(self.mol_list, mol_idx, ph_idx))

    @cached_property
    def norm(self):
        return super(MpProperty, self).norm

    @cached_property
    def ph_occupations(self):
        ph_occupations = []
        for imol, mol in enumerate(self.mol_list):
            for iph in range(len(mol.phs)):
                ph_occupations.append(self.calc_ph_occupation(imol, iph))
        return np.array(ph_occupations)

    @cached_property
    def e_occupations(self):
        return np.array([self.calc_e_occupation(i) for i in range(self.mol_num)])

    @cached_property
    def r_square(self):
        r_list = np.arange(0, self.mol_num)
        r_mean_square = np.average(r_list, weights=self.e_occupations) ** 2
        mean_r_square = np.average(r_list ** 2, weights=self.e_occupations)
        return mean_r_square - r_mean_square

    def invalidate_cache(self):
        for property in ['ph_occupations', 'e_occupations', 'r_square', 'norm']:
            if property in self.__dict__:
                del self.__dict__[property]

    @invalidate_cache_decorator
    def copy(self):
        return super(MpProperty, self).copy()

    @invalidate_cache_decorator
    def normalize(self, *args, **kwargs):
        return super(MpProperty, self).normalize(*args, **kwargs)

    def calc_energy(self, h_mpo):
        return self.expectation(h_mpo)

    def __str__(self):
        # too many digits in the default format
        e_occupations_str = ', '.join(['%.2f' % number for number in self.e_occupations])
        return 'Matrix product bond order:%s, electron occupations: %s' % (self.bond_dims, e_occupations_str)