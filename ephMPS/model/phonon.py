# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

from __future__ import absolute_import, print_function, unicode_literals

from collections import OrderedDict

import numpy as np

from ephMPS.utils import Quantity


class Phonon(object):
    '''
    phonon class has property:
    frequency : omega{}
    PES displacement: dis
    highest occupation levels: nlevels
    '''

    @classmethod
    def simple_phonon(cls, omega, displacement, n_phys_dim):
        complete_omega = [omega, omega]
        complete_displacement = [Quantity.zero(), displacement]
        return cls(complete_omega, complete_displacement, n_phys_dim)

    def __init__(self, omega, displacement, n_phys_dim, force3rd=None, nqboson=1, qbtrunc=0.0):
        # omega is a dictionary for different PES omega[0], omega[1]...
        self.omega = [o.as_au() for o in omega]
        # dis is a dictionary for different PES dis[0]=0.0, dis[1]...
        self.dis = [d.as_au() for d in displacement]

        if force3rd is None:
            self.force3rd = {}
            for i in range(len(omega)):
                self.force3rd[i] = 0.0
        else:
            self.force3rd = force3rd

        self.n_phys_dim = n_phys_dim
        self.nqboson = nqboson
        self.qbtrunc = qbtrunc
        self.base = int(round(n_phys_dim ** (1. / nqboson)))

    def to_dict(self):
        info_dict = OrderedDict()
        info_dict['omega'] = self.omega
        info_dict['displacement'] = self.dis
        info_dict['num physical dimension'] = self.n_phys_dim
        return info_dict

    @property
    def pbond(self):
        return [self.base] * self.nqboson

    @property
    def nlevels(self):
        return self.n_phys_dim

    """
    todo: These "term"s should be renamed by their physical meanings
    """
    @property
    def term10(self):
        return self.omega[1] ** 2 / np.sqrt(2. * self.omega[0]) * (- self.dis[1])

    @property
    def term11(self):
        return 3.0 * self.dis[1] ** 2 * self.force3rd[1] / np.sqrt(2. * self.omega[0])

    @property
    def term20(self):
        return 0.25 * (self.omega[1] ** 2 - self.omega[0] ** 2) / self.omega[0]


    @property
    def term21(self):
        return - 1.5 * self.dis[1] * self.force3rd[1] / self.omega[0]

    @property
    def term30(self):
        return self.force3rd[0] * (0.5 / self.omega[0]) ** 1.5

    @property
    def term31(self):
        return self.force3rd[1] * (0.5 / self.omega[0]) ** 1.5

    def printinfo(self):
        print("omega   = ", self.omega)
        print("displacement = ", self.dis)
        print("nlevels = ", self.n_phys_dim)
        print("nqboson = ", self.nqboson)
        print("qbtrunc = ", self.qbtrunc)
        print("base =", self.base)