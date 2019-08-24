# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

from collections import OrderedDict
from typing import List, Union

import numpy as np

from renormalizer.model.ephtable import EphTable
from renormalizer.model.mol import Mol, Phonon
from renormalizer.utils import Quantity
from renormalizer.utils.utils import cast_float


class MolList:

    def __init__(self, mol_list: List[Mol], j_matrix: Union[Quantity, np.ndarray, None], scheme: int=2, sbm=False):
        self.mol_list: List[Mol] = mol_list
        if sbm or j_matrix is None:
            assert len(self.mol_list) == 1
            j_matrix = Quantity(0)

        if isinstance(j_matrix, Quantity):
            self.j_matrix = construct_j_matrix(self.mol_num, j_matrix)
            self.j_constant = j_matrix
        else:
            self.j_matrix = j_matrix
            self.j_constant = None
        self.scheme = scheme

        self.ephtable = EphTable.from_mol_list(mol_list, scheme)
        self.pbond_list: List[int] = []
        if scheme < 4:
            if scheme == 3:
                assert self.check_nearest_neighbour()
            self._e_idx = []
            for mol in mol_list:
                self._e_idx.append(len(self.pbond_list))
                self.pbond_list.append(2)
                for ph in mol.dmrg_phs:
                    self.pbond_list += ph.pbond
        else:
            for imol, mol in enumerate(mol_list):
                if imol == len(mol_list) // 2:
                    self._e_idx = [len(self.pbond_list)] * len(mol_list)
                    self.pbond_list.append(len(mol_list))
                for ph in mol.dmrg_phs:
                    assert ph.is_simple
                    self.pbond_list += ph.pbond
        assert self.j_matrix.shape[0] == self.mol_num
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

    def switch_scheme(self, scheme):
        return self.__class__(self.mol_list, self.j_matrix, scheme)

    def e_idx(self, idx=0):
        return self._e_idx[idx]

    def ph_idx(self, eidx, phidx):
        if self.scheme < 4:
            start = self.e_idx(eidx)
            assert self.mol_list[eidx].no_qboson
            # skip the electron site
            return start + 1 + phidx
        elif self.scheme == 4:
            res = 0
            for mol in self.mol_list[:eidx]:
                assert mol.no_qboson
                res += mol.n_dmrg_phs
            if self.mol_num // 2 <= eidx:
                res += 1
            return res + phidx
        else:
            assert False

    def get_mpos(self, key, fun):
        if key not in self.mpos:
            mpos = fun(self)
            self.mpos[key] = mpos
        else:
            mpos = self.mpos[key]
        return mpos

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

    def check_nearest_neighbour(self):
        d = np.diag(np.diag(self.j_matrix, k=1), k=1)
        return np.allclose(d + d.T, self.j_matrix)

    def get_sub_mollist(self, span=None):
        assert self.mol_num % 2 == 1
        center_idx = self.mol_num // 2
        if span is None:
            span = self.mol_num // 10
        start_idx = center_idx-span
        end_idx = center_idx+span+1
        sub_list =self.mol_list[start_idx:end_idx]
        sub_matrix = self.j_matrix[start_idx:end_idx, start_idx:end_idx]
        return self.__class__(sub_list, sub_matrix, scheme=self.scheme), start_idx

    def get_pure_dmrg_mollist(self):
        l = []
        for mol in self.mol_list:
            mol = Mol(Quantity(mol.elocalex), mol.dmrg_phs, mol.dipole)
            l.append(mol)
        return self.__class__(l, self.j_matrix, scheme=self.scheme)

    def __getitem__(self, idx):
        return self.mol_list[idx]

    def __len__(self):
        return len(self.mol_list)

    def to_dict(self):
        info_dict = OrderedDict()
        info_dict["mol num"] = len(self)
        info_dict["electron phonon table"] = self.ephtable
        info_dict["mol list"] = [mol.to_dict() for mol in self.mol_list]
        if self.j_constant is None:
            info_dict["J matrix"] = cast_float(self.j_matrix)
        else:
            info_dict["J constant"] = self.j_constant.as_au()
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


def load_from_dict(param, scheme, lam: bool):
    temperature = Quantity(*param["temperature"])
    ph_list = [
        Phonon.simplest_phonon(
            Quantity(*omega), Quantity(*displacement), temperature=temperature, lam=lam
        )
        for omega, displacement in param["ph modes"]
    ]
    j_constant = Quantity(*param["j constant"])
    mol_list = MolList([Mol(Quantity(0), ph_list)] * param["mol num"],
                       j_constant, scheme=scheme
                       )
    return mol_list, temperature