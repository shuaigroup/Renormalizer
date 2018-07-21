# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

from __future__ import absolute_import, print_function, unicode_literals

import numpy as np

class Phonon(object):
    '''
    phonon class has property:
    frequency : omega{}
    PES displacement: dis
    highest occupation levels: nlevels
    '''

    def __init__(self, omega, displacement, nlevels, force3rd=None, nqboson=1, qbtrunc=0.0):
        # omega is a dictionary for different PES omega[0], omega[1]...
        self.omega = omega
        # dis is a dictionary for different PES dis[0]=0.0, dis[1]...
        self.dis = displacement

        if force3rd is None:
            self.force3rd = {}
            for i in range(len(omega)):
                self.force3rd[i] = 0.0
        else:
            self.force3rd = force3rd

        self.nlevels = nlevels
        self.nqboson = nqboson
        self.qbtrunc = qbtrunc
        self.base = int(round(nlevels ** (1. / nqboson)))

    @property
    def pbond(self):
        return [self.base] * self.nqboson

    """
    todo: These "term"s should be named by their phsycial meanings
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
        print("nlevels = ", self.nlevels)
        print("nqboson = ", self.nqboson)
        print("qbtrunc = ", self.qbtrunc)
        print("base =", self.base)