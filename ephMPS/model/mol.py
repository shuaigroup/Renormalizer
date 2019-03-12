# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
from __future__ import absolute_import, print_function, unicode_literals

from collections import OrderedDict
from typing import List

import numpy as np

from ephMPS.model.phonon import Phonon
from ephMPS.utils import Quantity


class Mol(object):
    """
    molecule class property:
    local excitation energy :  elocalex
    # of phonons : nphs
    condon dipole moment : dipole
    phonon information : ph
    """

    def __init__(self, elocalex, ph_list: List[Phonon], dipole=None):
        self.elocalex = elocalex.as_au()
        self.dipole = dipole
        self.dmrg_phs = tuple([ph for ph in ph_list if not ph.hartree])
        self.hartree_phs = tuple([ph for ph in ph_list if ph.hartree])

        def calc_lambda(phs):
            return sum([ph.reorganization_energy.as_au() for ph in phs])

        self.dmrg_e0 = calc_lambda(self.dmrg_phs)
        self.hartree_e0 = calc_lambda(self.hartree_phs)
        self.n_dmrg_phs = len(self.dmrg_phs)
        self.n_hartree_phs = len(self.hartree_phs)
        self.phhop = np.zeros([self.n_dmrg_phs, self.n_dmrg_phs])

    def create_phhop(self, phhopmat):
        self.phhop = phhopmat.copy()

    @property
    def pbond(self):
        pbond = [2]
        for ph in self.dmrg_phs:
            pbond += ph.pbond
        return pbond

    @property
    def reorganization_energy(self):
        return self.dmrg_e0 + self.hartree_e0

    @property
    def pure_dmrg(self):
        return not bool(self.hartree_phs)

    @property
    def pure_hartree(self):
        return not bool(self.dmrg_phs)

    def to_dict(self):
        info_dict = OrderedDict()
        info_dict["elocalex"] = self.elocalex
        info_dict["dipole"] = self.dipole
        info_dict["reorganization energy in a.u."] = self.reorganization_energy
        info_dict["dmrg phonon modes"] = self.n_dmrg_phs
        if self.n_hartree_phs:
            info_dict["dmrg phonon modes"] = self.n_dmrg_phs
        info_dict["DMRG phonon list"] = [ph.to_dict() for ph in self.dmrg_phs]
        if self.hartree_phs:
            info_dict["Hartree phonon list"] = [ph.to_dict() for ph in self.hartree_phs]
        return info_dict
