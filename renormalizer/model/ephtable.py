# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

electron = "e"

electrons = "es"

phonon = "ph"


class EphTable(tuple):
    @classmethod
    def all_phonon(cls, site_num):
        return cls([phonon] * site_num)

    @classmethod
    def from_mol_list(cls, mol_list, scheme):
        eph_list = []
        if scheme < 4:
            for mol in mol_list:
                eph_list.append(electron)
                for ph in mol.dmrg_phs:
                    eph_list.extend([phonon] * ph.nqboson)
        else:
            for imol, mol in enumerate(mol_list):
                if imol == len(mol_list) // 2:
                    eph_list.append(electrons)
                eph_list.extend([phonon] * mol.n_dmrg_phs)
                for ph in mol.dmrg_phs:
                    assert ph.is_simple
        return cls(eph_list)

    def is_electron(self, idx):
        # an electron site
        return self[idx] == electron

    def is_electrons(self, idx):
        # a site with all electron DOFs, used in scheme 4
        return self[idx] == electrons

    def is_phonon(self, idx):
        # a phonon site
        return self[idx] == phonon

    def __str__(self):
        return "[" + ", ".join(self) + "]"
