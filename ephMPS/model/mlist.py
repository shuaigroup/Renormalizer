# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

from collections import OrderedDict
from typing import List, Union

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

    def sub_mollist(self, span=None):
        assert self.mol_num % 2 == 1
        center_idx = self.mol_num // 2
        if span is None:
            span = self.mol_num // 10
        start_idx = center_idx-span
        end_idx = center_idx+span+1
        sub_list =self.mol_list[start_idx:end_idx]
        sub_matrix = self.j_matrix[start_idx:end_idx, start_idx:end_idx]
        return self.__class__(sub_list, sub_matrix), start_idx

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
