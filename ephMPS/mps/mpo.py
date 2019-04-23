from __future__ import absolute_import, print_function, unicode_literals

import logging

import numpy as np
import scipy

from ephMPS.model.ephtable import EphTable
from ephMPS.model import MolList
from ephMPS.mps.backend import xp
from ephMPS.mps.matrix import moveaxis, tensordot, ones
from ephMPS.mps.mp import MatrixProduct
from ephMPS.utils import Quantity
from ephMPS.utils.elementop import (
    construct_ph_op_dict,
    construct_e_op_dict,
    ph_op_matrix,
)
from ephMPS.utils.utils import roundrobin

logger = logging.getLogger(__name__)

# todo: refactor init
# the code is hard to understand...... need some closer look


def base_convert(n, base):
    """
    convert 10 base number to any base number
    """
    result = ""
    while True:
        tup = divmod(n, base)
        result += str(tup[1])
        if tup[0] == 0:
            return result[::-1]
        else:
            n = tup[0]


def get_pos(lidx, ridx, base, nqb):
    lstring = np.array(list(map(int, base_convert(lidx, base).zfill(nqb))))
    rstring = np.array(list(map(int, base_convert(ridx, base).zfill(nqb))))
    pos = tuple(roundrobin(lstring, rstring))
    return pos


def get_mpo_dim_qn(mol_list, scheme, rep):
    nmols = len(mol_list)
    mpo_dim = []
    mpo_qn = []
    if scheme == 1:
        for imol, mol in enumerate(mol_list):
            mpo_dim.append((imol + 1) * 2)
            mpo_qn.append([0] + [1, -1] * imol + [0])
            for iph in range(mol.n_dmrg_phs):
                if imol != nmols - 1:
                    mpo_dim.append((imol + 1) * 2 + 3)
                    mpo_qn.append([0, 0] + [1, -1] * (imol + 1) + [0])
                else:
                    mpo_dim.append(3)
                    mpo_qn.append([0, 0, 0])
    elif scheme == 2:
        # 0,1,2,3,4,5      3 is the middle
        # dim is 1*4, 4*6, 6*8, 8*6, 6*4, 4*1
        # 0,1,2,3,4,5,6    3 is the middle
        # dim is 1*4, 4*6, 6*8, 8*8, 8*6, 6*4, 4*1
        mididx = nmols // 2

        def elecdim(_imol):
            if _imol <= mididx:
                dim = (_imol + 1) * 2
            else:
                dim = (nmols - _imol + 1) * 2
            return dim

        for imol, mol in enumerate(mol_list):
            ldim = elecdim(imol)
            rdim = elecdim(imol + 1)

            mpo_dim.append(ldim)
            mpo_qn.append([0] + [1, -1] * (ldim // 2 - 1) + [0])
            for iph in range(mol.n_dmrg_phs):
                if rep == "chain":
                    if iph == 0:
                        mpo_dim.append(rdim + 1)
                        mpo_qn.append([0, 0] + [1, -1] * (rdim // 2 - 1) + [0])
                    else:
                        # replace the initial a^+a to b^+ and b
                        mpo_dim.append(rdim + 2)
                        mpo_qn.append([0, 0, 0] + [1, -1] * (rdim // 2 - 1) + [0])
                else:
                    mpo_dim.append(rdim + 1)
                    mpo_qn.append([0, 0] + [1, -1] * (rdim // 2 - 1) + [0])
    elif scheme == 3:
        # electronic nearest neighbor hopping
        # the electronic dimension is
        # 1*4, 4*4, 4*4,...,4*1
        for imol, mol in enumerate(mol_list):
            mpo_dim.append(4)
            mpo_qn.append([0, 1, -1, 0])
            for iph in range(mol.n_dmrg_phs):
                if imol != nmols - 1:
                    mpo_dim.append(5)
                    mpo_qn.append([0, 0, 1, -1, 0])
                else:
                    mpo_dim.append(3)
                    mpo_qn.append([0, 0, 0])
    else:
        raise ValueError(f"unknown scheme: {scheme}")
    mpo_dim[0] = 1
    return mpo_dim, mpo_qn


def get_qb_mpo_dim_qn(mol_list, old_dim, old_qn, rep):
    # quasi boson MPO dim
    qbopera = []  # b+b^\dagger MPO in quasi boson representation
    new_dim = []
    new_qn = []
    impo = 0
    for imol, mol in enumerate(mol_list):
        qbopera.append({})
        new_dim.append(old_dim[impo])
        new_qn.append(old_qn[impo])
        impo += 1
        for iph, ph in enumerate(mol.dmrg_phs):
            nqb = ph.nqboson
            if nqb != 1:
                if rep == "chain":
                    b = Mpo.quasi_boson("b", nqb, ph.qbtrunc, base=ph.base)
                    bdagger = Mpo.quasi_boson(
                        r"b^\dagger", nqb, ph.qbtrunc, base=ph.base
                    )
                    bpbdagger = Mpo.quasi_boson(
                        r"b + b^\dagger", nqb, ph.qbtrunc, base=ph.base
                    )
                    qbopera[imol]["b" + str(iph)] = b
                    qbopera[imol]["bdagger" + str(iph)] = bdagger
                    qbopera[imol]["bpbdagger" + str(iph)] = bpbdagger

                    if iph == 0:
                        if iph != mol.n_dmrg_phs - 1:
                            addmpodim = [
                                b[i].shape[0]
                                + bdagger[i].shape[0]
                                + bpbdagger[i].shape[0]
                                - 1
                                for i in range(nqb)
                            ]
                        else:
                            addmpodim = [bpbdagger[i].shape[0] - 1 for i in range(nqb)]
                        addmpodim[0] = 0
                    else:
                        addmpodim = [
                            (b[i].shape[0] + bdagger[i].shape[0]) * 2 - 2
                            for i in range(nqb)
                        ]
                        addmpodim[0] = 0

                else:
                    bpbdagger = Mpo.quasi_boson(
                        r"C1(b + b^\dagger) + C2(b + b^\dagger)^2",
                        nqb,
                        ph.qbtrunc,
                        base=ph.base,
                        c1=ph.term10,
                        c2=ph.term20,
                    )

                    qbopera[imol]["bpbdagger" + str(iph)] = bpbdagger
                    addmpodim = [i.shape[0] for i in bpbdagger]
                    addmpodim[0] = 0
                    # the first quasi boson MPO the row dim is as before, while
                    # the others the a_i^\dagger a_i should exist
            else:
                addmpodim = [0]

            # new MPOdim
            new_dim += [i + old_dim[impo] for i in addmpodim]
            # new MPOQN
            for iqb in range(nqb):
                new_qn.append(
                    old_qn[impo][0:1] + [0] * addmpodim[iqb] + old_qn[impo][1:]
                )
            impo += 1
    new_dim.append(1)
    new_qn[0] = [0]
    new_qn.append([0])
    # the boundary side of L/R side quantum number
    # MPOQN[:MPOQNidx] is L side
    # MPOQN[MPOQNidx+1:] is R side
    return qbopera, new_dim, new_qn


class Mpo(MatrixProduct):
    @classmethod
    def exact_propagator(cls, mol_list: MolList, x, space="GS", shift=0.0):
        """
        construct the GS space propagator e^{xH} exact MPO
        H=\\sum_{in} \\omega_{in} b^\\dagger_{in} b_{in}
        fortunately, the H is local. so e^{xH} = e^{xh1}e^{xh2}...e^{xhn}
        the bond dimension is 1
        shift is the a constant for H+shift
        """
        assert space in ["GS", "EX"]

        mpo = cls()
        if np.iscomplex(x):
            mpo.to_complex(inplace=True)
        mpo.mol_list = mol_list

        for imol, mol in enumerate(mol_list):
            if mol_list.scheme < 4:
                mo = np.eye(2).reshape(1, 2, 2, 1)
                mpo.append(mo)
            elif mol_list.scheme == 4:
                if len(mpo) == mol_list.e_idx():
                    n = mol_list.mol_num
                    mpo.append(np.eye(n).reshape(1, n, n, 1))
            else:
                assert False

            for ph in mol.dmrg_phs:

                if space == "EX":
                    # for the EX space, with quasiboson algorithm, the b^\dagger + b
                    # operator is not local anymore.
                    assert ph.nqboson == 1
                    ph_pbond = ph.pbond[0]
                    # construct the matrix exponential by diagonalize the matrix first
                    phop = construct_ph_op_dict(ph_pbond)

                    h_mo = (
                        phop[r"b^\dagger b"] * ph.omega[0]
                        + phop[r"(b^\dagger + b)^3"] * ph.term30
                        + phop[r"b^\dagger + b"] * (ph.term10 + ph.term11)
                        + phop[r"(b^\dagger + b)^2"] * (ph.term20 + ph.term21)
                        + phop[r"(b^\dagger + b)^3"] * (ph.term31 - ph.term30)
                    )

                    w, v = scipy.linalg.eigh(h_mo)
                    h_mo = np.diag(np.exp(x * w))
                    h_mo = v.dot(h_mo)
                    h_mo = h_mo.dot(v.T)
                    mo = h_mo.reshape(1, ph_pbond, ph_pbond, 1)

                    mpo.append(mo)

                elif space == "GS":
                    anharmo = False
                    # for the ground state space, yet doesn't support 3rd force
                    # potential quasiboson algorithm
                    ph_pbond = ph.pbond[0]
                    for i in range(len(ph.force3rd)):
                        anharmo = not np.allclose(
                            ph.force3rd[i] * ph.dis[i] / ph.omega[i], 0.0
                        )
                        if anharmo:
                            break
                    if not anharmo:
                        for iboson in range(ph.nqboson):
                            d = np.exp(
                                    x
                                    * ph.omega[0]
                                    * ph.base ** (ph.nqboson - iboson - 1)
                                    * np.arange(ph_pbond)
                                )
                            mo = np.diag(d).reshape(1, ph_pbond, ph_pbond, 1)
                            mpo.append(mo)
                    else:
                        assert ph.nqboson == 1
                        # construct the matrix exponential by diagonalize the matrix first
                        phop = construct_ph_op_dict(ph_pbond)
                        h_mo = (
                            phop[r"b^\dagger b"] * ph.omega[0]
                            + phop[r"(b^\dagger + b)^3"] * ph.term30
                        )
                        w, v = scipy.linalg.eigh(h_mo)
                        h_mo = np.diag(np.exp(x * w))
                        h_mo = v.dot(h_mo)
                        h_mo = h_mo.dot(v.T)

                        mo = np.zeros([1, ph_pbond, ph_pbond, 1])
                        mo[0, :, :, 0] = h_mo

                        mpo.append(mo)
                else:
                    assert False
        # shift the H by plus a constant

        mpo.qn = [[0]] * (len(mpo) + 1)
        mpo.qnidx = len(mpo) - 1
        mpo.qntot = 0

        # np.exp(shift * x) is usually very large
        mpo = mpo.scale(np.exp(shift * x), inplace=True)

        return mpo

    @classmethod
    def quasi_boson(cls, opera, nqb, trunc, base=2, c1=1.0, c2=1.0):
        """
        nqb : # of quasi boson sites
        opera : operator to be decomposed
                r"b + b^\\dagger"
        """
        assert opera in [
            r"b + b^\dagger",
            r"b^\dagger b",
            "b",
            r"b^\dagger",
            r"C1(b + b^\dagger) + C2(b + b^\dagger)^2",
        ]

        # the structure is [bra_highest_bit, ket_highest_bit,..., bra_lowest_bit,
        # ket_lowest_bit]
        mat = np.zeros([base] * nqb * 2)

        if opera == r"b + b^\dagger" or opera == r"b^\dagger" or opera == "b":
            if opera == r"b + b^\dagger" or opera == r"b^\dagger":
                for i in range(1, base ** nqb):
                    # b^+
                    pos = get_pos(i, i - 1, base, nqb)
                    mat[pos] = np.sqrt(i)

            if opera == r"b + b^\dagger" or opera == "b":
                for i in range(0, base ** nqb - 1):
                    # b
                    pos = get_pos(i, i + 1, base, nqb)
                    mat[pos] = np.sqrt(i + 1)

        elif opera == r"C1(b + b^\dagger) + C2(b + b^\dagger)^2":
            # b^+
            for i in range(1, base ** nqb):
                pos = get_pos(i, i - 1, base, nqb)
                mat[pos] = c1 * np.sqrt(i)
            # b
            for i in range(0, base ** nqb - 1):
                pos = get_pos(i, i + 1, base, nqb)
                mat[pos] = c1 * np.sqrt(i + 1)
            # bb
            for i in range(0, base ** nqb - 2):
                pos = get_pos(i, i + 2, base, nqb)
                mat[pos] = c2 * np.sqrt(i + 2) * np.sqrt(i + 1)
            # b^\dagger b^\dagger
            for i in range(2, base ** nqb):
                pos = get_pos(i, i - 2, base, nqb)
                mat[pos] = c2 * np.sqrt(i) * np.sqrt(i - 1)
            # b^\dagger b + b b^\dagger
            for i in range(0, base ** nqb):
                pos = get_pos(i, i, base, nqb)
                mat[pos] = c2 * float(i * 2 + 1)

        elif opera == r"b^\dagger b":
            # actually Identity operator can be constructed directly
            for i in range(0, base ** nqb):
                # I
                pos = get_pos(i, i, base, nqb)
                mat[pos] = float(i)

        # check the original mat
        # mat = np.moveaxis(mat,range(1,nqb*2,2),range(nqb,nqb*2))
        # print mat.reshape(base**nqb,base**nqb)

        # decompose canonicalise
        mpo = cls()
        mpo.ephtable = EphTable.all_phonon(nqb)
        mpo.pbond_list = [base] * nqb
        mpo.threshold = trunc
        mat = mat.reshape(1, -1)
        for idx in range(nqb - 1):
            u, s, vt = scipy.linalg.svd(
                mat.reshape(mat.shape[0] * base ** 2, -1), full_matrices=False
            )
            u = u.reshape(mat.shape[0], base, base, -1)
            mpo.append(u)
            mat = np.einsum("i, ij -> ij", s, vt)

        mpo.append(mat.reshape((-1, base, base, 1)))
        # print "original MPO shape:", [i.shape[0] for i in MPO] + [1]
        mpo.build_empty_qn()
        # compress
        mpo.canonicalise()
        mpo.compress()
        # print "trunc", trunc, "distance", mpslib.distance(MPO,MPOnew)
        # fidelity = mpslib.dot(mpslib.conj(MPOnew), MPO) / mpslib.dot(mpslib.conj(MPO), MPO)
        # print "compression fidelity:: ", fidelity
        # print "compressed MPO shape", [i.shape[0] for i in MPOnew] + [1]

        return mpo

    @classmethod
    def onsite(cls, mol_list: MolList, opera, dipole=False, mol_idx_set=None):
        assert opera in ["a", r"a^\dagger", r"a^\dagger a"]
        if mol_list.scheme == 4:
            return VirtualOnSite(mol_list, opera, dipole, mol_idx_set)
        nmols = len(mol_list)
        if mol_idx_set is None:
            mol_idx_set = set(np.arange(nmols))
        mpo_dim = []
        for imol in range(nmols):
            mpo_dim.append(2)
            for ph in mol_list[imol].dmrg_phs:
                for iboson in range(ph.nqboson):
                    if imol != nmols - 1:
                        mpo_dim.append(2)
                    else:
                        mpo_dim.append(1)

        mpo_dim[0] = 1
        mpo_dim.append(1)
        # print opera, "operator MPOdim", MPOdim

        mpo = cls()
        mpo.mol_list = mol_list
        impo = 0
        for imol in range(nmols):
            eop = construct_e_op_dict()
            mo = np.zeros([mpo_dim[impo], 2, 2, mpo_dim[impo + 1]])

            if imol in mol_idx_set:
                if dipole:
                    factor = mol_list[imol].dipole
                else:
                    factor = 1.0
            else:
                factor = 0.0

            mo[-1, :, :, 0] = factor * eop[opera]

            if imol != 0:
                mo[0, :, :, 0] = eop["Iden"]
            if imol != nmols - 1:
                mo[-1, :, :, -1] = eop["Iden"]
            mpo.append(mo)
            impo += 1

            for ph in mol_list[imol].dmrg_phs:
                for iboson in range(ph.nqboson):
                    pbond = mol_list.pbond_list[impo]
                    mo = np.zeros([mpo_dim[impo], pbond, pbond, mpo_dim[impo + 1]])
                    for ibra in range(pbond):
                        for idiag in range(mpo_dim[impo]):
                            mo[idiag, ibra, ibra, idiag] = 1.0

                    mpo.append(mo)
                    impo += 1

        # quantum number part
        # len(MPO)-1 = len(MPOQN)-2, the L-most site is R-qn
        mpo.qnidx = len(mpo) - 1

        totnqboson = 0
        for ph in mol_list[-1].dmrg_phs:
            totnqboson += ph.nqboson

        if opera == "a":
            mpo.qn = (
                [[0]]
                + [[-1, 0]] * (len(mpo) - totnqboson - 1)
                + [[-1]] * (totnqboson + 1)
            )
            mpo.qntot = -1
        elif opera == r"a^\dagger":
            mpo.qn = (
                [[0]]
                + [[1, 0]] * (len(mpo) - totnqboson - 1)
                + [[1]] * (totnqboson + 1)
            )
            mpo.qntot = 1
        elif opera == r"a^\dagger a":
            mpo.qn = (
                [[0]]
                + [[0, 0]] * (len(mpo) - totnqboson - 1)
                + [[0]] * (totnqboson + 1)
            )
            mpo.qntot = 0
        else:
            assert False
        mpo.qn[-1] = [0]

        return mpo

    @classmethod
    def ph_occupation_mpo(cls, mol_list: MolList, mol_idx: int, ph_idx=0):
        mpo = cls()
        mpo.mol_list = mol_list
        for imol, mol in enumerate(mol_list):
            if mol_list.scheme < 4:
                mpo.append(xp.eye(2).reshape(1, 2, 2, 1))
            elif mol_list.scheme == 4:
                if len(mpo) == mol_list.e_idx():
                    n = mol_list.mol_num
                    mpo.append(xp.eye(n).reshape(1, n, n, 1))
            else:
                assert False
            iph = 0
            for ph in mol.dmrg_phs:
                for iqph in range(ph.nqboson):
                    ph_pbond = ph.pbond[iqph]
                    if imol == mol_idx and iph == ph_idx:
                        mt = ph_op_matrix(r"b^\dagger b", ph_pbond)
                    else:
                        mt = ph_op_matrix("Iden", ph_pbond)
                    mpo.append(mt.reshape(1, ph_pbond, ph_pbond, 1))
                    iph += 1
        mpo.build_empty_qn()
        return mpo

    @classmethod
    def identity(cls, mol_list: MolList):
        mpo = cls()
        mpo.mol_list = mol_list
        for p in mol_list.pbond_list:
            mpo.append(xp.eye(p).reshape(1, p, p, 1))
        mpo.build_empty_qn()
        return mpo

    def _scheme4(self, mol_list: MolList, elocal_offset, offset):

        # setup some metadata
        self.rep = None
        self.use_dummy_qn = True
        self.offset = offset

        def get_marginal_phonon_mo(pdim, bdim, ph, phop):
            mo = np.zeros((1, pdim, pdim, bdim))
            mo[0, :, :, 0] = phop[r"b^\dagger b"] * ph.omega[0]
            mo[0, :, :, 1] = phop[r"b^\dagger + b"] * ph.term10
            mo[0, :, :, -1] = phop[r"Iden"]
            return mo

        def get_phonon_mo(pdim, bdim, ph, phop, isfirst):
            if isfirst:
                mo = np.zeros((bdim - 1, pdim, pdim, bdim))
            else:
                mo = np.zeros((bdim, pdim, pdim, bdim))
            mo[-1, :, :, 0] = phop[r"b^\dagger b"] * ph.omega[0]
            for i in range(bdim - 1):
                mo[i, :, :, i] = phop[r"Iden"]
            if isfirst:
                mo[bdim - 2, :, :, bdim - 2] = phop[r"b^\dagger + b"] * ph.term10
            else:
                mo[bdim - 1, :, :, bdim - 2] = phop[r"b^\dagger + b"] * ph.term10
            mo[-1, :, :, -1] = phop[r"Iden"]
            return mo

        nmol = mol_list.mol_num
        n_left_mol = nmol // 2
        n_right_mol = nmol - n_left_mol
        # the first half phonons
        for imol, mol in enumerate(mol_list[:n_left_mol]):
            for iph, ph in enumerate(mol.dmrg_phs):
                assert ph.is_simple
                pdim = ph.n_phys_dim
                bdim = imol + 3
                phop = construct_ph_op_dict(pdim)
                if imol == iph == 0:
                    mo = get_marginal_phonon_mo(pdim, bdim, ph, phop)
                    for i in range(mo.shape[1]):
                        mo[0, i, i, 0] -= offset.as_au()
                else:
                    mo = get_phonon_mo(pdim, bdim, ph, phop, iph == 0)
                self.append(mo)
        # the electronic part
        center_mo = np.zeros((n_left_mol+2, nmol, nmol, n_right_mol+2))
        center_mo[0, :, :, 0] = center_mo[-1, :, :, -1] = np.eye(nmol)
        j_matrix = mol_list.j_matrix.copy()
        for i in range(mol_list.mol_num):
            j_matrix[i, i] = mol_list[i].elocalex + mol_list[i].reorganization_energy
        if elocal_offset is not None:
            j_matrix += np.diag(elocal_offset)
        center_mo[-1, :, :, 0] = j_matrix
        for i in range(nmol):
            m = np.zeros((nmol, nmol))
            m[i, i] = 1
            if i < n_left_mol:
                center_mo[i+1, :, :, 0] = m
            else:
                center_mo[-1, :, :, i-n_left_mol+1] = m
        self.append(center_mo)
        # remaining phonons
        for imol, mol in enumerate(mol_list[n_left_mol:]):
            for iph, ph in enumerate(mol.dmrg_phs):
                assert ph.is_simple
                pdim = ph.n_phys_dim
                bdim = n_right_mol + 2 - imol
                phop = construct_ph_op_dict(pdim)
                if imol == n_right_mol - 1 and iph == mol.n_dmrg_phs - 1:
                    mo = get_marginal_phonon_mo(pdim, bdim, ph, phop)
                else:
                    islast =  iph == (mol.n_dmrg_phs - 1)
                    mo = get_phonon_mo(pdim, bdim, ph, phop, islast)
                self.append(mo.transpose((3, 1, 2, 0))[::-1, :, :, ::-1])
        self.build_empty_qn()

    def __init__(
        self,
        mol_list: MolList=None,
        rep="star",
        elocal_offset=None,
        offset=Quantity(0),
    ):
        """
        scheme 1: l to r
        scheme 2: l,r to middle, the bond dimension is smaller than scheme 1
        scheme 3: l to r, nearest neighbour exciton interaction
        rep (representation) has "star" or "chain"
        please see doc
        """
        assert rep in ["star", "chain", None]
        if rep is None:
            assert mol_list.scheme == 4

        super(Mpo, self).__init__()
        if mol_list is None:
            return
        if mol_list.pure_hartree:
            raise ValueError("Can't construct MPO for pure hartree model")

        # used in the hybrid TDDMRG/TDH algorithm
        if elocal_offset is not None:
            assert len(elocal_offset) == mol_list.mol_num

        self.mol_list = mol_list

        self.scheme = scheme = self.mol_list.scheme

        if scheme == 4:
            self._scheme4(mol_list, elocal_offset, offset)
            return

        self.rep = rep
        # offset of the hamiltonian, might be useful when doing td-hartree job
        self.offset = offset
        j_matrix = self.mol_list.j_matrix
        nmols = len(mol_list)


        mpo_dim, mpo_qn = get_mpo_dim_qn(mol_list, scheme, rep)

        qbopera, mpo_dim, self.qn = get_qb_mpo_dim_qn(mol_list, mpo_dim, mpo_qn, rep)

        self.qnidx = len(self.qn) - 2
        self.qntot = 0  # the total quantum number of each bond, for Hamiltonian it's 0

        # print "MPOdim", MPOdim

        # MPO
        impo = 0
        for imol, mol in enumerate(mol_list):

            mididx = nmols // 2

            # electronic part
            mo = np.zeros([mpo_dim[impo], 2, 2, mpo_dim[impo + 1]])
            eop = construct_e_op_dict()
            # last row operator
            elocal = mol.elocalex
            if elocal_offset is not None:
                elocal += elocal_offset[imol]
            mo[-1, :, :, 0] = eop[r"a^\dagger a"] * (elocal + mol.dmrg_e0)
            mo[-1, :, :, -1] = eop["Iden"]
            mo[-1, :, :, 1] = eop[r"a^\dagger a"]

            # first column operator
            if imol != 0:
                mo[0, :, :, 0] = eop["Iden"]
                if (scheme == 1) or (scheme == 2 and imol <= mididx):
                    for ileft in range(1, mpo_dim[impo] - 1):
                        if ileft % 2 == 1:
                            mo[ileft, :, :, 0] = (
                                eop["a"] * j_matrix[(ileft - 1) // 2, imol]
                            )
                        else:
                            mo[ileft, :, :, 0] = (
                                eop[r"a^\dagger"] * j_matrix[(ileft - 1) // 2, imol]
                            )
                elif scheme == 2 and imol > mididx:
                    mo[-3, :, :, 0] = eop["a"]
                    mo[-2, :, :, 0] = eop[r"a^\dagger"]
                elif scheme == 3:
                    mo[-3, :, :, 0] = eop["a"] * j_matrix[imol - 1, imol]
                    mo[-2, :, :, 0] = eop[r"a^\dagger"] * j_matrix[imol - 1, imol]

            # last row operator
            if imol != nmols - 1:
                if (scheme == 1) or (scheme == 2 and imol < mididx) or (scheme == 3):
                    mo[-1, :, :, -2] = eop["a"]
                    mo[-1, :, :, -3] = eop[r"a^\dagger"]
                elif scheme == 2 and imol >= mididx:
                    for jmol in range(imol + 1, nmols):
                        mo[-1, :, :, (nmols - jmol) * 2] = (
                            eop[r"a^\dagger"] * j_matrix[imol, jmol]
                        )
                        mo[-1, :, :, (nmols - jmol) * 2 + 1] = (
                            eop["a"] * j_matrix[imol, jmol]
                        )

            # mat body
            if imol != nmols - 1 and imol != 0:
                if scheme == 1 or (scheme == 2 and imol < mididx):
                    for ileft in range(2, 2 * (imol + 1)):
                        mo[ileft - 1, :, :, ileft] = eop["Iden"]
                elif scheme == 2 and imol > mididx:
                    for ileft in range(2, 2 * (nmols - imol)):
                        mo[ileft - 1, :, :, ileft] = eop["Iden"]
                elif scheme == 2 and imol == mididx:
                    for jmol in range(imol + 1, nmols):
                        for ileft in range(imol):
                            mo[ileft * 2 + 1, :, :, (nmols - jmol) * 2] = (
                                eop["Iden"] * j_matrix[ileft, jmol]
                            )
                            mo[ileft * 2 + 2, :, :, (nmols - jmol) * 2 + 1] = (
                                eop["Iden"] * j_matrix[ileft, jmol]
                            )
            # scheme 3 no body mat

            if imol == 0:
                for i in range(mo.shape[1]):
                    mo[0, i, i, 0] -= offset.as_au()
            self.append(mo)
            impo += 1

            # # of electronic operators retained in the phonon part, only used in
            # Mpo algorithm
            if rep == "chain":
                # except E and a^\dagger a
                nIe = mpo_dim[impo] - 2

            # phonon part
            for iph, ph in enumerate(mol.dmrg_phs):
                nqb = mol.dmrg_phs[iph].nqboson
                if nqb == 1:
                    pbond = self.pbond_list[impo]
                    phop = construct_ph_op_dict(pbond)
                    mo = np.zeros([mpo_dim[impo], pbond, pbond, mpo_dim[impo + 1]])
                    # first column
                    mo[0, :, :, 0] = phop["Iden"]
                    mo[-1, :, :, 0] = (
                        phop[r"b^\dagger b"] * ph.omega[0]
                        + phop[r"(b^\dagger + b)^3"] * ph.term30
                    )
                    if rep == "chain" and iph != 0:
                        mo[1, :, :, 0] = phop["b"] * mol.phhop[iph, iph - 1]
                        mo[2, :, :, 0] = phop[r"b^\dagger"] * mol.phhop[iph, iph - 1]
                    else:
                        mo[1, :, :, 0] = (
                            phop[r"b^\dagger + b"] * (ph.term10 + ph.term11)
                            + phop[r"(b^\dagger + b)^2"] * (ph.term20 + ph.term21)
                            + phop[r"(b^\dagger + b)^3"] * (ph.term31 - ph.term30)
                        )
                    if imol != nmols - 1 or iph != mol.n_dmrg_phs - 1:
                        mo[-1, :, :, -1] = phop["Iden"]
                        if rep == "chain":
                            if iph == 0:
                                mo[-1, :, :, 1] = phop[r"b^\dagger"]
                                mo[-1, :, :, 2] = phop["b"]
                                for icol in range(3, mpo_dim[impo + 1] - 1):
                                    mol[icol - 1, :, :, icol] = phop("Iden")
                            elif iph == mol.n_dmrg_phs - 1:
                                for icol in range(1, mpo_dim[impo + 1] - 1):
                                    mo[icol + 2, :, :, icol] = phop["Iden"]
                            else:
                                mo[-1, :, :1] = phop[r"b^\dagger"]
                                mo[-1, :, :, 2] = phop["b"]
                                for icol in range(3, mpo_dim[impo + 1] - 1):
                                    mo[icol, :, :, icol] = phop["Iden"]
                        elif rep == "star":
                            if iph != mol.n_dmrg_phs - 1:
                                for icol in range(1, mpo_dim[impo + 1] - 1):
                                    mo[icol, :, :, icol] = phop["Iden"]
                            else:
                                for icol in range(1, mpo_dim[impo + 1] - 1):
                                    mo[icol + 1, :, :, icol] = phop["Iden"]
                    self.append(mo)
                    impo += 1
                else:
                    # b + b^\dagger in Mpo representation
                    for iqb in range(nqb):
                        pbond = self.pbond_list[impo]
                        phop = construct_ph_op_dict(pbond)
                        mo = np.zeros([mpo_dim[impo], pbond, pbond, mpo_dim[impo + 1]])

                        if rep == "star":
                            bpbdagger = qbopera[imol]["bpbdagger" + str(iph)][
                                iqb
                            ].asnumpy()

                            mo[0, :, :, 0] = phop["Iden"]
                            mo[-1, :, :, 0] = (
                                phop[r"b^\dagger b"]
                                * mol.dmrg_phs[iph].omega[0]
                                * float(mol.dmrg_phs[iph].base) ** (nqb - iqb - 1)
                            )

                            #  the # of identity operator
                            if iqb != nqb - 1:
                                nI = mpo_dim[impo + 1] - bpbdagger.shape[-1] - 1
                            else:
                                nI = mpo_dim[impo + 1] - 1

                            for iset in range(1, nI + 1):
                                mo[-iset, :, :, -iset] = phop["Iden"]

                            # b + b^\dagger
                            if iqb != nqb - 1:
                                mo[
                                    1 : bpbdagger.shape[0] + 1,
                                    :,
                                    :,
                                    1 : bpbdagger.shape[-1] + 1,
                                ] = bpbdagger
                            else:
                                mo[
                                    1 : bpbdagger.shape[0] + 1,
                                    :,
                                    :,
                                    0 : bpbdagger.shape[-1],
                                ] = bpbdagger

                        elif rep == "chain":

                            b = qbopera[imol]["b" + str(iph)][iqb]
                            bdagger = qbopera[imol]["bdagger" + str(iph)][iqb]
                            bpbdagger = qbopera[imol]["bpbdagger" + str(iph)][iqb]

                            mo[0, :, :0] = phop["Iden"]
                            mo[-1, :, :, 0] = (
                                phop[r"b^\dagger b"]
                                * mol.dmrg_phs[iph].omega[0]
                                * float(mol.dmrg_phs[iph].base) ** (nqb - iqb - 1)
                            )

                            #  the # of identity operator
                            if impo == len(mpo_dim) - 2:
                                nI = nIe - 1
                            else:
                                nI = nIe

                            # print
                            # "nI", nI
                            for iset in range(1, nI + 1):
                                mo[-iset, :, :, -iset] = phop["Iden"]

                            if iph == 0:
                                # b + b^\dagger
                                if iqb != nqb - 1:
                                    mo[
                                        1 : bpbdagger.shape[0] + 1,
                                        :,
                                        :,
                                        1 : bpbdagger.shape[-1] + 1,
                                    ] = bpbdagger
                                else:
                                    mo[1 : bpbdagger.shape[0] + 1, :, :, 0:1] = (
                                        bpbdagger * ph.term10
                                    )
                            else:
                                # b^\dagger, b
                                if iqb != nqb - 1:
                                    mo[
                                        1 : b.shape[0] + 1, :, :, 1 : b.shape[-1] + 1
                                    ] = b
                                    mo[
                                        b.shape[0]
                                        + 1 : b.shape[0]
                                        + 1
                                        + bdagger.shape[0],
                                        :,
                                        :,
                                        b.shape[-1]
                                        + 1 : b.shape[-1]
                                        + 1
                                        + bdagger.shape[-1],
                                    ] = bdagger
                                else:
                                    mo[1 : b.shape[0] + 1, :, :, 0:1] = (
                                        b * mol.phhop[iph, iph - 1]
                                    )
                                    mo[
                                        b.shape[0]
                                        + 1 : b.shape[0]
                                        + 1
                                        + bdagger.shape[0],
                                        :,
                                        :,
                                        0:1,
                                    ] = (bdagger * mol.phhop[iph, iph - 1])

                            if iph != mol.n_dmrg_phs - 1:
                                if iph == 0:
                                    loffset = bpbdagger.shape[0]
                                    roffset = bpbdagger.shape[-1]
                                else:
                                    loffset = b.shape[0] + bdagger.shape[0]
                                    roffset = b.shape[-1] + bdagger.shape[-1]
                                    # b^\dagger, b
                                if iqb == 0:
                                    mo[
                                        -1:,
                                        :,
                                        :,
                                        roffset + 1 : roffset + 1 + bdagger.shape[-1],
                                    ] = bdagger
                                    mo[
                                        -1:,
                                        :,
                                        :,
                                        roffset
                                        + 1
                                        + bdagger.shape[-1] : roffset
                                        + 1
                                        + bdagger.shape[-1]
                                        + b.shape[-1],
                                    ] = b
                                elif iqb == nqb - 1:
                                    # print
                                    # "He", loffset + 1, \
                                    # loffset + 1 + bdagger.shape[0], loffset + 1 + bdagger.shape[0] + b.shape[0],
                                    mo[
                                        loffset + 1 : loffset + 1 + bdagger.shape[0],
                                        :,
                                        :,
                                        1:2,
                                    ] = bdagger
                                    mo[
                                        loffset
                                        + 1
                                        + bdagger.shape[0] : loffset
                                        + 1
                                        + bdagger.shape[0]
                                        + b.shape[0],
                                        :,
                                        :,
                                        2:3,
                                    ] = b
                                else:
                                    mo[
                                        loffset + 1 : loffset + 1 + bdagger.shape[0],
                                        :,
                                        :,
                                        roffset + 1 : roffset + 1 + bdagger.shape[-1],
                                    ] = bdagger
                                    mo[
                                        loffset
                                        + 1
                                        + bdagger.shape[0] : loffset
                                        + 1
                                        + bdagger.shape[0]
                                        + b.shape[0],
                                        :,
                                        :,
                                        roffset
                                        + 1
                                        + bdagger.shape[-1] : roffset
                                        + 1
                                        + bdagger.shape[-1]
                                        + b.shape[-1],
                                    ] = b

                        self.append(mo)
                        impo += 1

    def _get_sigmaqn(self, idx):
        if self.ephtable.is_electron(idx):
            return np.array([0, -1, 1, 0])
        else:
            return np.array([0] * self.pbond_list[idx] ** 2)

    @property
    def is_mps(self):
        return False

    @property
    def is_mpo(self):
        return True

    @property
    def is_mpdm(self):
        return False

    def metacopy(self):
        new = super().metacopy()
        # some mpo may not have these things
        attrs = ["scheme", "rep", "offset"]
        for attr in attrs:
            if hasattr(self, attr):
                setattr(new, attr, getattr(self, attr))
        return new

    @property
    def dummy_qn(self):
        return [[0] * dim for dim in self.bond_dims]

    @property
    def digest(self):
        return np.array([mt.var() for mt in self]).var()

    def promote_mt_type(self, mp):
        if self.is_complex and not mp.is_complex:
            mp.to_complex(inplace=True)
        return mp

    def apply(self, mp, canonicalise=False) -> MatrixProduct:
        # todo: use meta copy to save time, could be subtle when complex type is involved
        # todo: inplace version (saved memory and can be used in `hybrid_exact_propagator`)
        new_mps = self.promote_mt_type(mp.copy())
        if mp.is_mps:
            # mpo x mps
            for i, (mt_self, mt_other) in enumerate(zip(self, mp)):
                assert mt_self.shape[2] == mt_other.shape[1]
                # mt=np.einsum("apqb,cqd->acpbd",mpo[i],mps[i])
                mt = xp.moveaxis(
                    tensordot(mt_self.array, mt_other.array, axes=([2], [1])), 3, 1
                )
                mt = mt.reshape(
                    (
                        mt_self.shape[0] * mt_other.shape[0],
                        mt_self.shape[1],
                        mt_self.shape[-1] * mt_other.shape[-1],
                    )
                )
                new_mps[i] = mt
        elif mp.is_mpo or mp.is_mpdm:
            # mpo x mpo
            for i, (mt_self, mt_other) in enumerate(zip(self, mp)):
                assert mt_self.shape[2] == mt_other.shape[1]
                # mt=np.einsum("apqb,cqrd->acprbd",mt_s,mt_o)
                mt = xp.moveaxis(
                    tensordot(mt_self.array, mt_other.array, axes=([2], [1])),
                    [-3, -2],
                    [1, 3],
                )
                mt = mt.reshape(
                    (
                        mt_self.shape[0] * mt_other.shape[0],
                        mt_self.shape[1],
                        mt_other.shape[2],
                        mt_self.shape[-1] * mt_other.shape[-1],
                    )
                )
                new_mps[i] = mt
        else:
            assert False
        orig_idx = new_mps.qnidx
        new_mps.move_qnidx(self.qnidx)
        qn = self.qn if not mp.use_dummy_qn else self.dummy_qn
        new_mps.qn = [
            np.add.outer(np.array(qn_o), np.array(qn_m)).ravel().tolist()
            for qn_o, qn_m in zip(qn, new_mps.qn)
        ]
        new_mps.qntot += self.qntot
        new_mps.move_qnidx(orig_idx)
        new_mps.set_peak_bytes()
        # concerns about whether to canonicalise:
        # * canonicalise helps to keep mps in a truly canonicalised state
        # * canonicalise comes with a cost. Unnecessary canonicalise (for example in P&C evolution and
        #   expectation calculation) hampers performance.
        if canonicalise:
            new_mps.canonicalise()
        return new_mps

    def contract(self, mps):

        """
        mapply->canonicalise->compress
        """
        new_mps = self.apply(mps)
        new_mps.canonicalise()
        new_mps.compress()
        return new_mps

    def conj_trans(self):
        new_mpo = self.metacopy()
        for i in range(new_mpo.site_num):
            new_mpo[i] = moveaxis(self[i], (1, 2), (2, 1)).conj()
        new_mpo.qn = [[-i for i in mt_qn] for mt_qn in new_mpo.qn]
        return new_mpo

    def full_operator(self):
        dim = np.prod(self.pbond_list)
        if 20000 < dim:
            raise ValueError("operator too large")
        res = ones((1, 1, 1, 1))
        for mt in self:
            dim1 = res.shape[1] * mt.shape[1]
            dim2 = res.shape[2] * mt.shape[2]
            dim3 = mt.shape[-1]
            res = tensordot(res, mt, axes=1).transpose((0, 1, 3, 2, 4, 5)).reshape(1, dim1, dim2, dim3)
        return res[0, :, :, 0]


class VirtualOnSite(Mpo):
    """
    acts like a mpo but manipulates mps or mpdm directly
    """

    def __init__(self, mol_list, opera, dipole=False, mol_idx_set=None):
        super().__init__(None)
        assert mol_list.scheme == 4
        self.mol_list = mol_list
        self.type = "onsite"
        self.opera = opera
        assert not dipole  # not implemented
        self.dipole = dipole
        if mol_idx_set is None:
            self.mol_idx_set = np.arange(len(mol_list))
        else:
            self.mol_idx_set = mol_idx_set
        assert len(self.mol_idx_set) != 0

    def apply(self, mp: MatrixProduct, canonicalise=False):
        assert mp.mol_list.scheme == 4
        new_mp = mp.copy()
        e_ms = mp[self.mol_list.e_idx()]
        new_ms = xp.zeros_like(e_ms.array)
        if self.opera == r"a^\dagger a":
            for idx in self.mol_idx_set:
                new_ms[:, idx, ...] = e_ms[:, idx, ...]
        elif self.opera == r"a^\dagger":
            assert (e_ms.original_shape[0], e_ms.original_shape[-1]) == (1, 1)
            for idx in self.mol_idx_set:
                if mp.is_mps:
                    new_ms[0, idx, 0] = 1
                elif mp.is_mpdm:
                    new_ms[0, idx, idx, 0] = 1
                else:
                    assert False
        else:
            assert False
        new_mp[self.mol_list.e_idx()] = new_ms

        if canonicalise:
            new_mp.canonicalise()
        return new_mp