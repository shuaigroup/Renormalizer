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

    def __init__(self, mol_list: List[Mol], j_matrix: Union[Quantity, np.ndarray, None], scheme: int = 2, periodic: bool = False):
        self.period = periodic
        self.mol_list: List[Mol] = mol_list

        # construct the electronic coupling matrix
        if j_matrix is None:
            # spin-boson model
            assert len(self.mol_list) == 1
            j_matrix = Quantity(0)

        if isinstance(j_matrix, Quantity):
            self.j_matrix = construct_j_matrix(self.mol_num, j_matrix, periodic)
            self.j_constant = j_matrix
        else:
            if periodic:
                assert j_matrix[0][-1] != 0 and j_matrix[-1][0] != 0
            self.j_matrix = j_matrix
            self.j_constant = None
        self.scheme = scheme

        assert self.j_matrix.shape[0] == self.mol_num

        self.ephtable = EphTable.from_mol_list(mol_list, scheme)
        self.pbond_list: List[int] = []

        if scheme < 4:
            # the order is e0,v00,v01,...,e1,v10,v11,...
            if scheme == 3:
                assert self.check_nearest_neighbour()
            self._e_idx = []
            for mol in mol_list:
                self._e_idx.append(len(self.pbond_list))
                self.pbond_list.append(2)
                for ph in mol.dmrg_phs:
                    self.pbond_list += ph.pbond
        else:
            # the order is v00,v01,..,v10,v11,...,e0/e1/e2,...,v30,v31...
            for imol, mol in enumerate(mol_list):
                if imol == len(mol_list) // 2:
                    self._e_idx = [len(self.pbond_list)] * len(mol_list)
                    self.pbond_list.append(len(mol_list)+1)
                for ph in mol.dmrg_phs:
                    assert ph.is_simple
                    self.pbond_list += ph.pbond

        # reusable mpos for the system
        self.mpos = dict()

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
        return self.__class__(self.mol_list.copy(), self.j_matrix.copy(), scheme, self.period)

    def copy(self):
        return self.switch_scheme(self.scheme)

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

    def check_nearest_neighbour(self):
        d = np.diag(np.diag(self.j_matrix, k=1), k=1)
        d = d + d.T
        d[0, -1] = self.j_matrix[-1, 0]
        d[-1, 0] = self.j_matrix[0, -1]
        return np.allclose(d, self.j_matrix)

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


def construct_j_matrix(mol_num, j_constant, periodic):
    # nearest neighbour interaction
    j_constant_au = j_constant.as_au()
    j_list = np.ones(mol_num - 1) * j_constant_au
    j_matrix = np.diag(j_list, k=-1) + np.diag(j_list, k=1)
    if periodic:
        j_matrix[-1, 0] = j_matrix[0, -1] = j_constant_au
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
