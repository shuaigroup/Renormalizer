# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
from __future__ import absolute_import, print_function, unicode_literals


import numpy as np

from ephMPS.mps.elementop import construct_e_op_dict


class Mol(object):
    '''
    molecule class property:
    local excitation energy :  elocalex
    # of phonons : nphs
    condon dipole moment : dipole
    phonon information : ph
    '''

    def __init__(self, elocalex, dipole, ph_list):
        self.elocalex = elocalex
        self.dipole = dipole
        self.nphs = len(ph_list)
        self.phs = []
        self.e0 = 0.0
        for ph in ph_list:
            self.phs.append(ph)
            self.e0 += 0.5 * ph.omega[1] ** 2 * ph.dis[1] ** 2 - ph.dis[1] ** 3 * ph.force3rd[1]
        self.phhop = np.zeros([self.nphs, self.nphs])

    def create_phhop(self, phhopmat):
        self.phhop = phhopmat.copy()

    def printinfo(self):
        print("local excitation energy = ", self.elocalex)
        print("nphs = ", self.nphs)
        print("dipole = ", self.dipole)

    @property
    def pbond(self):
        pbond = [2]
        for ph in self.phs:
            pbond += ph.pbond
        return pbond
