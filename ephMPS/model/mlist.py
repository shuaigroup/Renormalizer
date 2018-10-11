# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

from collections import OrderedDict

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

    @property
    def ph_modes_num(self):
        return sum([mol.nphs for mol in self.mol_list])

    def __getitem__(self, item):
        return self.mol_list[item]

    def __len__(self):
        return len(self.mol_list)

    def to_dict(self):
        info_dict = OrderedDict()
        info_dict['mol num'] = len(self)
        info_dict['electron phonon table'] = self.ephtable
        info_dict['physical bond list'] = self.pbond_list
        info_dict['mol list'] = [mol.to_dict() for mol in self.mol_list]
        return info_dict



