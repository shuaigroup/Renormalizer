# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

from collections import OrderedDict
from typing import List

import numpy as np

from renormalizer.model.phonon import Phonon
from renormalizer.utils import Quantity


class Mol:
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
        if len(ph_list) == 0:
            raise ValueError("No phonon mode in phonon list")
        self.ph_list = ph_list
        self.e0 = sum([ph.reorganization_energy.as_au() for ph in ph_list])

    @property
    def reorganization_energy(self):
        return self.e0

    @property
    def gs_zpe(self):
        e = 0.
        for ph in self.ph_list:
            e += ph.omega[0]
        return e/2
    
    @property
    def ex_zpe(self):
        e = 0.
        for ph in self.ph_list:
            e += ph.omega[1]
        return e/2

    def to_dict(self):
        info_dict = OrderedDict()
        info_dict["elocalex"] = self.elocalex
        info_dict["dipole"] = self.dipole
        info_dict["reorganization energy in a.u."] = self.reorganization_energy
        info_dict["phonon list"] = [ph.to_dict() for ph in self.ph_list]
        return info_dict

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)
