# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

from ephMPS.model.ephtable import EphTable


class MolList(object):

    def __init__(self, mol_list):
        self.mol_list = mol_list
        self.ephtable = EphTable.from_mol_list(mol_list)
        self.pbond_list = []
        for mol in mol_list:
            self.pbond_list += mol.pbond

    @property
    def mol_num(self):
        return len(self.mol_list)

    def __getitem__(self, item):
        return self.mol_list[item]

    def __len__(self):
        return len(self.mol_list)


