# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

from __future__ import division, print_function, absolute_import

import logging
import datetime

import numpy as np
try:
    import seaborn as sns
except ImportError:
    logging.warn('Seaborn not installed, draw module down')
    sns = None
try:
    from matplotlib import pyplot as plt
except ImportError:
    logging.warn('Matplotlib not installed, draw module down')
    plt = None

from ephMPS.utils import pickle, MpAdaptor, TdMpsJob
from ephMPS import constant
from ephMPS.mps import Mps, Mpo


logger = logging.getLogger(__name__)


class CtMps(MpAdaptor):

    def __init__(self, mps):
        super(CtMps, self).__init__(mps)
        self.e_occupations = [self.calc_e_occupation(i) for i in range(self.mol_num)]
        self.ph_occupations = None
        self.r_square = self.calc_r_square()

    def calc_e_occupation(self, idx):
        return self.expectation(Mpo.onsite(self.mol_list, 'a^\dagger a', site_idx_set={idx}))

    def calc_ph_occupation(self, idx):
        pass

    def calc_r_square(self):
        r_list = np.arange(0, self.mol_num)
        r_square = np.average(r_list ** 2, weights=self.e_occupations) - np.average(r_list, weights=self.e_occupations) ** 2
        return r_square

    def __str__(self):
        return 'ct'


class ChargeTransport(TdMpsJob):

    def __init__(self, mol_list, j_constant, temperature=0):
        self.mol_list = mol_list
        self.j_constant = j_constant
        self.temperature = temperature
        self.mpo = None
        super(ChargeTransport, self).__init__()

    @property
    def mol_num(self):
        return self.mol_list.mol_num

    def create_electron(self, gs_mp):
        creation_operator = Mpo.onsite(self.mol_list, 'a^\dagger', site_idx_set={self.mol_num // 2})
        return creation_operator.apply(gs_mp)

    def init_mps(self):
        j_matrix = construct_j_matrix(self.mol_num, self.j_constant)
        self.mpo = Mpo(self.mol_list, j_matrix, scheme=3)
        gs_mp = Mps.gs(self.mol_list)
        if 0 < self.temperature:
            gs_mp = Mpo.from_mps(gs_mp).thermal_prop(inplace=True)
        init_mp = CtMps(self.create_electron(gs_mp))
        return init_mp

    def evolve_single_step(self, evolve_dt):
        return self.latest_mps.evolve(self.mpo, evolve_dt)

    @property
    def r_square_array(self):
        return np.array([mps.r_square for mps in self.tdmps_list])

    @property
    def occupations_array(self):
        return np.array([mps.occupations for mps in self.tdmps_list])


def construct_j_matrix(mol_num, j_constant):
    j_matrix = np.zeros((mol_num, mol_num))
    for i in range(mol_num):
        for j in range(mol_num):
            if i - j == 1 or i - j == -1:
                j_matrix[i][j] = j_constant
    return j_matrix / constant.au2ev
