# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

from collections import OrderedDict
from typing import List

import numpy as np

from renormalizer.model.phonon import Phonon
from renormalizer.utils import Quantity


def create_heatbath(heatbath_energy, method="ohmic") -> List[Phonon]:
    modes = 200
    if method == "ohmic":
        s = 0.5
        omega_c = Quantity(100, "cm-1").as_au()
        start = Quantity(10, "cm-1").as_au()
        end = 6 * omega_c
        omegas = np.arange(start, end, (end - start) / modes)
        spectral_density = omega_c ** (1 - s) * omegas ** s * np.exp(-omegas / omega_c)
        hr_factors = spectral_density / (omegas ** 2)
    elif method == "debye":
        omega_c = 2 * heatbath_energy
        start = Quantity(10, "cm-1").as_au()
        end = 5 * omega_c
        omegas = np.arange(start, end, (end-start) / modes)
        spectral_density = omegas * omega_c / (omegas ** 2 + omega_c ** 2)
        hr_factors = spectral_density / (omegas ** 2)
    elif method == "test":
        start = Quantity(490, "cm-1").as_au()
        end = Quantity(510, "cm-1").as_au()
        omegas = np.arange(start, end, (end-start) / modes)
        hr_factors = np.ones_like(omegas)
    else:
        raise ValueError(f'unknown method: {method}')
    total_energy = np.sum(hr_factors * omegas)
    hr_factors *= heatbath_energy / total_energy
    displacements = np.sqrt(2 * hr_factors / omegas)
    return [Phonon.simplest_phonon(Quantity(omega), Quantity(displacement), hartree=True) for omega, displacement in zip(omegas, displacements)]


class Mol:
    """
    molecule class property:
    local excitation energy :  elocalex
    # of phonons : nphs
    condon dipole moment : dipole
    phonon information : ph
    """

    def __init__(self, elocalex, ph_list: List[Phonon], dipole=None, heatbath=False, tunnel=Quantity(0)):
        self.elocalex = elocalex.as_au()
        self.dipole = dipole
        self.tunnel = tunnel.as_au()
        if len(ph_list) == 0:
            raise ValueError("No phonon mode in phonon list")
        self.dmrg_phs = [ph for ph in ph_list if not ph.hartree]
        self.hartree_phs = [ph for ph in ph_list if ph.hartree]

        def calc_lambda(phs):
            return sum([ph.reorganization_energy.as_au() for ph in phs])

        if heatbath:
            #fraction = 10
            #heatbath_energy = (calc_lambda(self.dmrg_phs) + calc_lambda(self.hartree_phs)) * fraction
            heatbath_energy = Quantity(50, "cm-1").as_au()
            self.hartree_phs += create_heatbath(heatbath_energy)

        self.dmrg_e0 = calc_lambda(self.dmrg_phs)
        self.hartree_e0 = calc_lambda(self.hartree_phs)
        self.n_dmrg_phs = len(self.dmrg_phs)
        self.n_hartree_phs = len(self.hartree_phs)

    @property
    def reorganization_energy(self):
        return self.dmrg_e0 + self.hartree_e0

    @property
    def pure_dmrg(self):
        return not bool(self.hartree_phs)

    @property
    def pure_hartree(self):
        return not bool(self.dmrg_phs)

    @property
    def no_qboson(self):
        for ph in self.dmrg_phs:
            if ph.nqboson != 1:
                return False
        return True

    @property
    def phs(self):
        return self.dmrg_phs + self.hartree_phs

    @property
    def sbm(self):
        return self.tunnel != 0

    @property
    def gs_zpe(self):
        e = 0.
        for ph in self.dmrg_phs:
            e += ph.omega[0]
        return e/2
    
    @property
    def ex_zpe(self):
        e = 0.
        for ph in self.dmrg_phs:
            e += ph.omega[1]
        return e/2

    def to_dict(self):
        info_dict = OrderedDict()
        info_dict["elocalex"] = self.elocalex
        info_dict["dipole"] = self.dipole
        info_dict["reorganization energy in a.u."] = self.reorganization_energy
        info_dict["tunnel"] = self.tunnel
        info_dict["dmrg phonon modes"] = self.n_dmrg_phs
        if self.n_hartree_phs:
            info_dict["dmrg phonon modes"] = self.n_dmrg_phs
        info_dict["DMRG phonon list"] = [ph.to_dict() for ph in self.dmrg_phs]
        if self.hartree_phs:
            info_dict["Hartree phonon list"] = [ph.to_dict() for ph in self.hartree_phs]
        return info_dict

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)
