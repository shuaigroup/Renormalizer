# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

electron = 'e'

phonon = 'ph'


class EphTable(list):

    def __init__(self, mol_list):
        super(EphTable, self).__init__()
        for mol in mol_list:
            self.append(electron)
            for ph in mol.phs:
                self.extend([phonon] * ph.nqboson)

    def is_electron(self, idx):
        return self[idx] == electron

    def is_phonon(self, idx):
        return self[idx] == phonon

    def __str__(self):
        return '[' + ', '.join(self) + ']'

