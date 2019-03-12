# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

electron = "e"

phonon = "ph"


class EphTable(tuple):
    @classmethod
    def all_phonon(cls, site_num):
        return cls([phonon] * site_num)

    @classmethod
    def from_mol_list(cls, mol_list):
        eph_list = []
        for mol in mol_list:
            eph_list.append(electron)
            for ph in mol.dmrg_phs:
                eph_list.extend([phonon] * ph.nqboson)
        return cls(eph_list)

    def electron_idx(self, idx):
        for res, st in enumerate(self):
            if st == electron:
                idx -= 1
            if idx == -1:
                return res

    def is_electron(self, idx):
        return self[idx] == electron

    def is_phonon(self, idx):
        return self[idx] == phonon

    def get_sigmaqn(self, idx):
        pass

    @property
    def num_electron_site(self):
        res = 0
        for i in self:
            if i == electron:
                res += 1
        return res

    def __str__(self):
        return "[" + ", ".join(self) + "]"
