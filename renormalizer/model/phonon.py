# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

from collections import OrderedDict
from typing import List

import numpy as np
from scipy.stats import binom

from renormalizer.utils import Quantity
from renormalizer.utils.elementop import construct_ph_op_dict


def all_positive_or_all_negative(array):
    if (np.logical_or(array <= 0, np.isclose(array, np.zeros_like(array)))).all():
        return True
    elif (np.logical_or(0 <= array, np.isclose(array, np.zeros_like(array)))).all():
        return True
    else:
        return False

class Phonon(object):
    """
    phonon class has property:
    frequency : omega{}
    PES displacement: dis
    highest occupation levels: nlevels
    """

    @classmethod
    def simplest_phonon(cls, omega, displacement, temperature: Quantity = Quantity(0), lam: bool=False, max_pdim=128):
        # detect pdim automatically
        if lam:
            # the second argument is lambda
            d = np.sqrt(2 * displacement.as_au()) / omega.as_au()
            displacement = Quantity(d)
        pdim = 256
        while True:
            trial_phonon = cls.simple_phonon(omega, displacement, pdim)
            evecs = trial_phonon.get_displacement_evecs()
            gs = evecs[:, 0]
            assert all_positive_or_all_negative(gs)
            if 0.9999 < gs[:len(gs) // 2].sum() / gs.sum():
                # too many levels
                pdim //= 2
            elif 0.001 < np.abs(gs[-1]):
                if pdim == 256:
                    # not enough levels. the required pdim is insanely large
                    # something went wrong!
                    raise ValueError(f"Too many phonon level required. omega: {omega}. displacement: {displacement}")
                else:
                    # a too large step
                    pdim *= 2
                    break
            else:
                break
        t = temperature.as_au()
        thermal_dim = int(t * 10 / omega.as_au())
        pdim = min(pdim + thermal_dim, max_pdim)
        return cls.simple_phonon(omega, displacement, pdim)


    @classmethod
    def simple_phonon(cls, omega, displacement, n_phys_dim):
        complete_omega = [omega, omega]
        complete_displacement = [Quantity(0), displacement]
        return cls(complete_omega, complete_displacement, n_phys_dim)

    def __init__(
        self,
        omega,
        displacement,
        n_phys_dim: int =None,
    ):
        # omega is a list for different PES omega[0], omega[1]...
        self.omega = [o.as_au() for o in omega]
        # dis is a list for different PES dis[0]=0.0, dis[1]...
        self.dis = [d.as_au() for d in displacement]

        self.n_phys_dim: int = n_phys_dim


    def get_displacement_evecs(self) -> np.ndarray:
        h = np.zeros((self.n_phys_dim, self.n_phys_dim))
        for i in range(self.n_phys_dim):
            h[i, i] = i
        off_diag = np.zeros((self.n_phys_dim, self.n_phys_dim))
        g = self.coupling_constant
        for i in range(self.n_phys_dim - 1):
            # note displacement is defined as negative
            off_diag[i + 1, i] = -g * np.sqrt(i + 1)
        h = h + off_diag + off_diag.T
        evals, evecs = np.linalg.eigh(h)
        return evecs

    def split(self, n=2, width: Quantity=Quantity(10, "cm-1")) -> List["Phonon"]:
        assert self.is_simple
        rv = binom(n-1, 0.5)
        width = width.as_au()
        step = 2 * width / (n - 1)
        omegas = np.linspace(self.omega[0] - width, self.omega[0] + width + step, n)
        phonons = []
        for i, omega in enumerate(omegas):
            lam = rv.pmf(i) * self.reorganization_energy
            ph = Phonon.simplest_phonon(Quantity(omega), lam, lam=True)
            phonons.append(ph)
        return phonons

    def to_dict(self):
        info_dict = OrderedDict()
        info_dict["omega"] = self.omega
        info_dict["displacement"] = self.dis
        info_dict["num physical dimension"] = self.n_phys_dim
        return info_dict

    @property
    def pbond(self):
        return self.n_phys_dim

    @property
    def nlevels(self):
        return self.n_phys_dim

    @property
    def reorganization_energy(self):
        dis_diff = self.dis[1] - self.dis[0]
        return Quantity(
            0.5 * dis_diff ** 2 * self.omega[1] ** 2
        )

    @property
    def e0(self):
        # an alias for reorganization energy
        return self.reorganization_energy

    @property
    def is_simple(self):
        return self.omega[0] == self.omega[1]

    @property
    def coupling_constant(self):  # the $g$
        return float(np.sqrt(self.reorganization_energy.as_au() / self.omega[0]))

    @property
    def term10(self):
        return self.omega[1] ** 2 / np.sqrt(2.0 * self.omega[0]) * (-self.dis[1])

    def printinfo(self):
        print("omega   = ", self.omega)
        print("displacement = ", self.dis)
        print("nlevels = ", self.n_phys_dim)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
