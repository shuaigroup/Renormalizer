# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

from cached_property import cached_property
import numpy as np

from ephMPS.mps.mp import MatrixProduct
from ephMPS.mps.mpobase import MpoBase
from ephMPS.utils.utils import sizeof_fmt


cached_property_set = set()

def _cached_property(func):
    cached_property_set.add(func.__name__)
    return cached_property(func)

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

    @_cached_property
    def norm(self):
        return super(MpProperty, self).norm

    @_cached_property
    def ph_occupations(self):
        ph_occupations = []
        for imol, mol in enumerate(self.mol_list):
            for iph in range(len(mol.dmrg_phs)):
                ph_occupations.append(self.calc_ph_occupation(imol, iph))
        return np.array(ph_occupations)

    @_cached_property
    def e_occupations(self):
        return np.array([self.calc_e_occupation(i) for i in range(self.mol_num)])

    @_cached_property
    def r_square(self):
        r_list = np.arange(0, self.mol_num)
        r_mean_square = np.average(r_list, weights=self.e_occupations) ** 2
        mean_r_square = np.average(r_list ** 2, weights=self.e_occupations)
        return mean_r_square - r_mean_square

    def invalidate_cache(self):
        for property in cached_property_set:
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

    def clear_memory(self):
        # make a cache
        for prop in cached_property_set:
            _ = getattr(self, prop)
        self.clear()

    def __str__(self):
        # too many digits in the default format
        e_occupations_str = ', '.join(['%.2f' % number for number in self.e_occupations])
        template_str = 'threshold: {:g}, current size: {}, peak size: {}, Matrix product bond order:{}, electron occupations: {}'
        return template_str.format(self.threshold, sizeof_fmt(self.total_bytes), sizeof_fmt(self.peak_bytes), self.bond_dims, e_occupations_str)