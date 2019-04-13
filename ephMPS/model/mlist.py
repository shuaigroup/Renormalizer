# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

from collections import OrderedDict
from typing import List, Union
import random

import numpy as np

from ephMPS.model.ephtable import EphTable
from ephMPS.model.mol import Mol
from ephMPS.utils import Quantity


class MolList(object):
    def __init__(self, mol_list: List[Mol], j_matrix: Union[Quantity, np.ndarray]):
        self.mol_list: List[Mol] = mol_list
        if isinstance(j_matrix, Quantity):
            self.j_matrix = construct_j_matrix(self.mol_num, j_matrix)
            self.j_constant = j_matrix
        else:
            self.j_matrix = j_matrix
            self.j_constant = None
        self.ephtable = EphTable.from_mol_list(mol_list)
        assert self.j_matrix.shape[0] == self.ephtable.num_electron_site
        self.pbond_list = []
        for mol in mol_list:
            self.pbond_list += mol.pbond
        # reusable mpos for the system
        self.mpos = dict()

        # originally designed for symmetry enforced evolution. Not implemented
        # need symmetric MPO but we currently don't have one.
        self.is_symmetric = self.check_symmetric()

        self.fluctuation_coeffs = None

    @property
    def mol_num(self):
        return len(self.mol_list)

    @property
    def ph_modes_num(self):
        return sum([mol.n_dmrg_phs for mol in self.mol_list])

    @property
    def pure_dmrg(self):
        for mol in self:
            if not mol.pure_dmrg:
                return False
        return True

    @property
    def pure_hartree(self):
        for mol in self:
            if not mol.pure_hartree:
                return False
        return True

    @property
    def elocalex_array(self):
        return np.array([m.elocalex for m in self.mol_list])

    @property
    def adjacent_transfer_integral(self):
        return np.diag(self.j_matrix, 1)

    def check_symmetric(self):
        # first check for j matrix
        rot = np.rot90(self.j_matrix)
        if not np.allclose(rot, rot.T):
            return False
        # then check for mols
        for i in range(len(self.mol_list) // 2):
            if self.mol_list[i] != self.mol_list[-i]:
                return False
        return True

    def get_sub_mollist(self, span=None):
        assert self.mol_num % 2 == 1
        center_idx = self.mol_num // 2
        if span is None:
            span = self.mol_num // 10
        start_idx = center_idx-span
        end_idx = center_idx+span+1
        sub_list =self.mol_list[start_idx:end_idx]
        sub_matrix = self.j_matrix[start_idx:end_idx, start_idx:end_idx]
        return self.__class__(sub_list, sub_matrix), start_idx

    def get_pure_dmrg_mollist(self):
        l = []
        for mol in self.mol_list:
            mol = Mol(Quantity(mol.elocalex), mol.dmrg_phs, mol.dipole)
            l.append(mol)
        return self.__class__(l, self.j_matrix)


    def get_fluctuation_mollist(self, time):
        if self.fluctuation_coeffs is None:
            self.fluctuation_coeffs = []
            for i in range(self.mol_num - 1):
                n = 10
                coeffs = (np.random.rand(n) - 0.5, (np.random.rand(n) - 0.5) * np.pi)
                self.fluctuation_coeffs.append(coeffs)
        # site_energy
        j_list = []
        for mol, (a, b) in zip(self.mol_list, self.fluctuation_coeffs):
            # e = np.sin(a*time + b).sum()
            # by experiment, e has std of 2 and cycle of 10
            # what we want is std of 20 meV, cycle of 50 cm-1 (27000 a.u.)
            e = np.sin(a * (time / 2700) + b).sum() * 10
            j_list.append(Quantity(e, "meV").as_au())
        j_matrix = np.zeros_like(self.j_matrix)
        for i in range(self.mol_num - 1):
            j_matrix[i+1, i] = self.j_matrix[i+1, i] + j_list[i]
        j_matrix = j_matrix + j_matrix.T
        new_mollist = self.__class__(self.mol_list, j_matrix)
        # mpos can still be reused
        new_mollist.mpos = self.mpos
        return new_mollist

    def get_fluctuation_mollist4(self, time):
        if self.fluctuation_coeffs is None:
            self.fluctuation_coeffs = []
            for i in range(self.mol_num):
                n = 10
                coeffs = (np.random.rand(n) - 0.5, (np.random.rand(n) - 0.5) * np.pi)
                self.fluctuation_coeffs.append(coeffs)
        # site_energy
        l = []
        for mol, (a, b) in zip(self.mol_list, self.fluctuation_coeffs):
            # e = np.sin(a*time + b).sum()
            # by experiment, e has std of 2 and cycle of 10
            # what we want is std of 0.2 eV, cycle of 200 cm-1 (7000 a.u.)
            #e = np.sin(a * (time / 700) + b).sum() / 10
            e = np.sin(a * (time / 70) + b).sum() / 10
            new_mol = Mol(Quantity(e, "eV"), mol.phs)
            l.append(new_mol)
        new_mollist = self.__class__(l, self.j_matrix)
        # mpos can still be reused
        new_mollist.mpos = self.mpos
        return new_mollist

    def get_fluctuation_mollist3(self):
        # site_energy
        l = []
        for mol in self.mol_list:
            new_mol = Mol(Quantity(random.random() * 0.5, "eV"), mol.phs)
            l.append(new_mol)
        new_mollist = self.__class__(l, self.j_matrix)
        # mpos can still be reused
        new_mollist.mpos = self.mpos
        return new_mollist

    def get_fluctuation_mollist2(self):
        # transfer integral
        j_matrix: np.ndarray = self.j_matrix.copy()
        fluctuation = np.random.random(j_matrix.shape)
        j_matrix *= fluctuation
        # make j_matrix symmetric
        j_matrix = (j_matrix + j_matrix.T) / 2
        new_mollist = self.__class__(self.mol_list, j_matrix)
        # mpos can still be reused
        new_mollist.mpos = self.mpos
        return new_mollist

    def __getitem__(self, idx):
        return self.mol_list[idx]

    def __len__(self):
        return len(self.mol_list)

    def to_dict(self):
        info_dict = OrderedDict()
        info_dict["mol num"] = len(self)
        info_dict["electron phonon table"] = self.ephtable
        info_dict["physical bond list"] = self.pbond_list
        info_dict["mol list"] = [mol.to_dict() for mol in self.mol_list]
        return info_dict


def construct_j_matrix(mol_num, j_constant):
    # nearest neighbour interaction
    j_constant_au = j_constant.as_au()
    j_matrix = np.zeros((mol_num, mol_num))
    for i in range(mol_num):
        for j in range(mol_num):
            if i - j == 1 or i - j == -1:
                j_matrix[i][j] = j_constant_au
    return j_matrix
