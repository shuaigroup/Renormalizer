# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

electron = 'e'

phonon = 'ph'


class EphTable(list):

    @classmethod
    def all_phonon(cls, site_num):
        return cls([phonon] * site_num)

    @classmethod
    def from_mol_list(cls, mol_list):
        ephtable = cls()
        for mol in mol_list:
            ephtable.append(electron)
            for ph in mol.phs:
                ephtable.extend([phonon] * ph.nqboson)
        return ephtable


    def is_electron(self, idx):
        return self[idx] == electron

    def is_phonon(self, idx):
        return self[idx] == phonon

    def __str__(self):
        return '[' + ', '.join(self) + ']'

