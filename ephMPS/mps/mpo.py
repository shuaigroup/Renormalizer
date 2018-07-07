from __future__ import absolute_import, print_function, unicode_literals

import numpy as np
import scipy

from ephMPS.mps.matrix import MatrixOp
from ephMPS.mps.mp import MatrixProduct
from ephMPS.mps.ephtable import EphTable
from ephMPS.mps.elementop import EElementOpera, PhElementOpera
from ephMPS.utils.utils import roundrobin


def baseConvert(n, base):
    '''
    convert 10 base number to any base number
    '''
    result = ''
    while True:
        tup = divmod(n, base)
        result += str(tup[1])
        if tup[0] == 0:
            return result[::-1]
        else:
            n = tup[0]


class Mpo(MatrixProduct):
    
    @classmethod
    def Quasi_Boson(self, opera, nqb, trunc, base=2, C1=1.0, C2=1.0):
        '''
        nqb : # of quasi boson sites
        opera : operator to be decomposed
                "b + b^\dagger"
        '''
        assert opera in ["b + b^\dagger", "b^\dagger b", "b", "b^\dagger",
                         "C1(b + b^\dagger) + C2(b + b^\dagger)^2"]

        # the structure is [bra_highest_bit, ket_highest_bit,..., bra_lowest_bit,
        # ket_lowest_bit]
        mat = np.zeros([base, ] * nqb * 2)

        if opera == "b + b^\dagger" or opera == "b^\dagger" or opera == "b":
            if opera == "b + b^\dagger" or opera == "b^\dagger":
                for i in range(1, base ** nqb):
                    # b^+
                    lstring = np.array(map(int, baseConvert(i, base).zfill(nqb)))
                    rstring = np.array(map(int, baseConvert(i - 1, base).zfill(nqb)))
                    pos = tuple(roundrobin(lstring, rstring))
                    mat[pos] = np.sqrt(i)

            if opera == "b + b^\dagger" or opera == "b":
                for i in range(0, base ** nqb - 1):
                    # b
                    lstring = np.array(map(int, baseConvert(i, base).zfill(nqb)))
                    rstring = np.array(map(int, baseConvert(i + 1, base).zfill(nqb)))
                    pos = tuple(roundrobin(lstring, rstring))
                    mat[pos] = np.sqrt(i + 1)

        elif opera == "C1(b + b^\dagger) + C2(b + b^\dagger)^2":
            # b^+
            for i in range(1, base ** nqb):
                lstring = np.array(map(int, baseConvert(i, base).zfill(nqb)))
                rstring = np.array(map(int, baseConvert(i - 1, base).zfill(nqb)))
                pos = tuple(roundrobin(lstring, rstring))
                mat[pos] = C1 * np.sqrt(i)
            # b
            for i in range(0, base ** nqb - 1):
                lstring = np.array(map(int, baseConvert(i, base).zfill(nqb)))
                rstring = np.array(map(int, baseConvert(i + 1, base).zfill(nqb)))
                pos = tuple(roundrobin(lstring, rstring))
                mat[pos] = C1 * np.sqrt(i + 1)
            # bb
            for i in range(0, base ** nqb - 2):
                lstring = np.array(map(int, baseConvert(i, base).zfill(nqb)))
                rstring = np.array(map(int, baseConvert(i + 2, base).zfill(nqb)))
                pos = tuple(roundrobin(lstring, rstring))
                mat[pos] = C2 * np.sqrt(i + 2) * np.sqrt(i + 1)
            # b^\dagger b^\dagger
            for i in range(2, base ** nqb):
                lstring = np.array(map(int, baseConvert(i, base).zfill(nqb)))
                rstring = np.array(map(int, baseConvert(i - 2, base).zfill(nqb)))
                pos = tuple(roundrobin(lstring, rstring))
                mat[pos] = C2 * np.sqrt(i) * np.sqrt(i - 1)
            # b^\dagger b + b b^\dagger
            for i in range(0, base ** nqb):
                lstring = np.array(map(int, baseConvert(i, base).zfill(nqb)))
                rstring = np.array(map(int, baseConvert(i, base).zfill(nqb)))
                pos = tuple(roundrobin(lstring, rstring))
                mat[pos] = C2 * float(i * 2 + 1)

        elif opera == "b^\dagger b":
            # actually Identity operator can be constructed directly
            for i in range(0, base ** nqb):
                # I
                lstring = np.array(map(int, baseConvert(i, base).zfill(nqb)))
                rstring = np.array(map(int, baseConvert(i, base).zfill(nqb)))
                pos = tuple(roundrobin(lstring, rstring))
                mat[pos] = float(i)

        # check the original mat
        # mat = np.moveaxis(mat,range(1,nqb*2,2),range(nqb,nqb*2))
        # print mat.reshape(base**nqb,base**nqb)

        # decompose canonicalise
        MPO = Mpo()
        mat = mat.reshape(1, -1)
        for idx in range(nqb - 1):
            U, S, Vt = scipy.linalg.svd(mat.reshape(mat.shape[0] * base ** 2, -1),
                                        full_matrices=False)
            U = U.reshape(mat.shape[0], base, base, -1)
            MPO.append(U)
            mat = np.einsum("i, ij -> ij", S, Vt)

        MPO.append(mat.reshape(-1, base, base, 1))
        # print "original MPO shape:", [i.shape[0] for i in MPO] + [1]

        # compress
        MPOnew = MPO.compress('l', trunc=trunc)
        # print "trunc", trunc, "distance", mpslib.distance(MPO,MPOnew)
        # fidelity = mpslib.dot(mpslib.conj(MPOnew), MPO) / mpslib.dot(mpslib.conj(MPO), MPO)
        # print "compression fidelity:: ", fidelity
        # print "compressed MPO shape", [i.shape[0] for i in MPOnew] + [1]

        return MPOnew

    def __init__(self, mol_list=None, J_matrix=None, scheme=2, rep="star", elocal_offset=None):
        '''
        scheme 1: l to r
        scheme 2: l,r to middle, the bond dimension is smaller than scheme 1
        scheme 3: l to r, nearest neighbour exciton interaction 
        rep (representation) has "star" or "chain"
        please see doc
        '''
        assert rep in ["star", "chain"]

        super(Mpo, self).__init__()
        self.mtype = MatrixOp
        if mol_list is None or J_matrix is None:
            return

        MPOdim = []
        nmols = len(mol_list)
        MPOQN = []

        self.ephtable = EphTable(mol_list)

        pbond_list = []
        for mol in mol_list:
            pbond_list += mol.pbond



        # used in the hybrid TDDMRG/TDH algorithm
        if elocal_offset is not None:
            assert len(elocal_offset) == nmols

        # MPOdim  
        if scheme == 1:
            for imol in range(nmols):
                MPOdim.append((imol + 1) * 2)
                MPOQN.append([0] + [1, -1] * imol + [0])
                for iph in range(mol_list[imol].nphs):
                    if imol != nmols - 1:
                        MPOdim.append((imol + 1) * 2 + 3)
                        MPOQN.append([0, 0] + [1, -1] * (imol + 1) + [0])
                    else:
                        MPOdim.append(3)
                        MPOQN.append([0, 0, 0])
        elif scheme == 2:
            # 0,1,2,3,4,5      3 is the middle 
            # dim is 1*4, 4*6, 6*8, 8*6, 6*4, 4*1 
            # 0,1,2,3,4,5,6    3 is the middle 
            # dim is 1*4, 4*6, 6*8, 8*8, 8*6, 6*4, 4*1 
            mididx = nmols // 2

            def elecdim(imol):
                if imol <= mididx:
                    dim = (imol + 1) * 2
                else:
                    dim = (nmols - imol + 1) * 2
                return dim

            for imol in range(nmols):
                ldim = elecdim(imol)
                rdim = elecdim(imol + 1)

                MPOdim.append(ldim)
                MPOQN.append([0] + [1, -1] * (ldim // 2 - 1) + [0])
                for iph in range(mol_list[imol].nphs):
                    if rep == "chain":
                        if iph == 0:
                            MPOdim.append(rdim + 1)
                            MPOQN.append([0, 0] + [1, -1] * (rdim // 2 - 1) + [0])
                        else:
                            # replace the initial a^+a to b^+ and b
                            MPOdim.append(rdim + 2)
                            MPOQN.append([0, 0, 0] + [1, -1] * (rdim // 2 - 1) + [0])
                    else:
                        MPOdim.append(rdim + 1)
                        MPOQN.append([0, 0] + [1, -1] * (rdim // 2 - 1) + [0])
        elif scheme == 3:
            # electronic nearest neighbor hopping
            # the electronic dimension is
            # 1*4, 4*4, 4*4,...,4*1
            for imol in range(nmols):
                MPOdim.append(4)
                MPOQN.append([0, 1, -1, 0])
                for iph in range(mol_list[imol].nphs):
                    if imol != nmols - 1:
                        MPOdim.append(5)
                        MPOQN.append([0, 0, 1, -1, 0])
                    else:
                        MPOdim.append(3)
                        MPOQN.append([0, 0, 0])

        MPOdim[0] = 1

        # quasi boson MPO dim
        qbopera = []  # b+b^\dagger MPO in quasi boson representation
        MPOdimnew = []
        MPOQNnew = []
        impo = 0
        for imol in range(nmols):
            qbopera.append({})
            MPOdimnew.append(MPOdim[impo])
            MPOQNnew.append(MPOQN[impo])
            impo += 1
            for iph in range(mol_list[imol].nphs):
                nqb = mol_list[imol].phs[iph].nqboson
                if nqb != 1:
                    if rep == "chain":
                        b = Mpo.Quasi_Boson("b", nqb,
                                            mol_list[imol].phs[iph].qbtrunc, base=mol_list[imol].phs[iph].base)
                        bdagger = Mpo.Quasi_Boson("b^\dagger", nqb,
                                                  mol_list[imol].phs[iph].qbtrunc, base=mol_list[imol].phs[iph].base)
                        bpbdagger = Mpo.Quasi_Boson("b + b^\dagger", nqb,
                                                    mol_list[imol].phs[iph].qbtrunc, base=mol_list[imol].phs[iph].base)
                        qbopera[imol]["b" + str(iph)] = b
                        qbopera[imol]["bdagger" + str(iph)] = bdagger
                        qbopera[imol]["bpbdagger" + str(iph)] = bpbdagger

                        if iph == 0:
                            if iph != mol_list[imol].nphs - 1:
                                addmpodim = [b[i].shape[0] + bdagger[i].shape[0] + bpbdagger[i].shape[0] - 1 for i in
                                             range(nqb)]
                            else:
                                addmpodim = [bpbdagger[i].shape[0] - 1 for i in range(nqb)]
                            addmpodim[0] = 0
                        else:
                            addmpodim = [(b[i].shape[0] + bdagger[i].shape[0]) * 2 - 2 for i in range((nqb))]
                            addmpodim[0] = 0

                    else:
                        bpbdagger = Mpo.Quasi_Boson("C1(b + b^\dagger) + C2(b + b^\dagger)^2", nqb,
                                                    mol_list[imol].phs[iph].qbtrunc,
                                                    base=mol_list[imol].phs[iph].base,
                                                    C1=mol_list[imol].phs[iph].omega[1] ** 2 / np.sqrt(
                                                                   2. * mol_list[imol].phs[iph].omega[0]) * -
                                                                  mol_list[imol].phs[iph].dis[1],
                                                    C2=0.25 * (mol_list[imol].phs[iph].omega[1] ** 2 -
                                                                          mol_list[imol].phs[iph].omega[0] ** 2) /
                                                                  mol_list[imol].phs[iph].omega[0])

                        qbopera[imol]["bpbdagger" + str(iph)] = bpbdagger
                        addmpodim = [i.shape[0] for i in bpbdagger]
                        addmpodim[0] = 0
                        # the first quasi boson MPO the row dim is as before, while
                        # the others the a_i^\dagger a_i should exist
                else:
                    addmpodim = [0]

                # new MPOdim
                MPOdimnew += [i + MPOdim[impo] for i in addmpodim]
                # new MPOQN
                for iqb in range(nqb):
                    MPOQNnew.append(MPOQN[impo][0:1] + [0] * addmpodim[iqb] + MPOQN[impo][1:])
                impo += 1

        # print "original MPOdim", MPOdim + [1]

        MPOdim = MPOdimnew
        MPOQN = MPOQNnew

        MPOdim.append(1)
        MPOQN[0] = [0]
        MPOQN.append([0])
        # the boundary side of L/R side quantum number
        # MPOQN[:MPOQNidx] is L side
        # MPOQN[MPOQNidx+1:] is R side
        MPOQNidx = len(MPOQN) - 2
        MPOQNtot = 0  # the total quantum number of each bond, for Hamiltonian it's 0              

        # print "MPOdim", MPOdim

        # MPO
        impo = 0
        for imol in range(nmols):

            mididx = nmols // 2

            # electronic part
            mpo = np.zeros([MPOdim[impo], pbond_list[impo], pbond_list[impo], MPOdim[impo + 1]])
            for ibra in range(pbond_list[impo]):
                for iket in range(pbond_list[impo]):
                    # last row operator
                    elocal = mol_list[imol].elocalex
                    if elocal_offset is not None:
                        elocal += elocal_offset[imol]
                    mpo[-1, ibra, iket, 0] = EElementOpera("a^\dagger a", ibra, iket) \
                                             * (elocal + mol_list[imol].e0)
                    mpo[-1, ibra, iket, -1] = EElementOpera("Iden", ibra, iket)
                    mpo[-1, ibra, iket, 1] = EElementOpera("a^\dagger a", ibra, iket)

                    # first column operator
                    if imol != 0:
                        mpo[0, ibra, iket, 0] = EElementOpera("Iden", ibra, iket)
                        if (scheme == 1) or (scheme == 2 and imol <= mididx):
                            for ileft in range(1, MPOdim[impo] - 1):
                                if ileft % 2 == 1:
                                    mpo[ileft, ibra, iket, 0] = EElementOpera("a", ibra, iket) * J_matrix[
                                        (ileft - 1) // 2, imol]
                                else:
                                    mpo[ileft, ibra, iket, 0] = EElementOpera("a^\dagger", ibra, iket) * J_matrix[
                                        (ileft - 1) // 2, imol]
                        elif (scheme == 2 and imol > mididx):
                            mpo[-3, ibra, iket, 0] = EElementOpera("a", ibra, iket)
                            mpo[-2, ibra, iket, 0] = EElementOpera("a^\dagger", ibra, iket)
                        elif scheme == 3:
                            mpo[-3, ibra, iket, 0] = EElementOpera("a", ibra, iket) * J_matrix[imol - 1, imol]
                            mpo[-2, ibra, iket, 0] = EElementOpera("a^\dagger", ibra, iket) * J_matrix[imol - 1, imol]

                    # last row operator
                    if imol != nmols - 1:
                        if (scheme == 1) or (scheme == 2 and imol < mididx) or (scheme == 3):
                            mpo[-1, ibra, iket, -2] = EElementOpera("a", ibra, iket)
                            mpo[-1, ibra, iket, -3] = EElementOpera("a^\dagger", ibra, iket)
                        elif scheme == 2 and imol >= mididx:
                            for jmol in range(imol + 1, nmols):
                                mpo[-1, ibra, iket, (nmols - jmol) * 2] = EElementOpera("a^\dagger", ibra, iket) * J_matrix[
                                    imol, jmol]
                                mpo[-1, ibra, iket, (nmols - jmol) * 2 + 1] = EElementOpera("a", ibra, iket) * J_matrix[
                                    imol, jmol]

                    # mat body
                    if imol != nmols - 1 and imol != 0:
                        if (scheme == 1) or (scheme == 2 and (imol < mididx)):
                            for ileft in range(2, 2 * (imol + 1)):
                                mpo[ileft - 1, ibra, iket, ileft] = EElementOpera("Iden", ibra, iket)
                        elif (scheme == 2 and (imol > mididx)):
                            for ileft in range(2, 2 * (nmols - imol)):
                                mpo[ileft - 1, ibra, iket, ileft] = EElementOpera("Iden", ibra, iket)
                        elif (scheme == 2 and imol == mididx):
                            for jmol in range(imol + 1, nmols):
                                for ileft in range(imol):
                                    mpo[ileft * 2 + 1, ibra, iket, (nmols - jmol) * 2] = EElementOpera("Iden", ibra,
                                                                                                       iket) * J_matrix[
                                                                                             ileft, jmol]
                                    mpo[ileft * 2 + 2, ibra, iket, (nmols - jmol) * 2 + 1] = EElementOpera("Iden", ibra,
                                                                                                           iket) * J_matrix[
                                                                                                 ileft, jmol]
                                    # scheme 3 no body mat

            self.append(mpo)
            impo += 1

            # # of electronic operators retained in the phonon part, only used in
            # Mpo algorithm
            if rep == "chain":
                # except E and a^\dagger a
                nIe = MPOdim[impo] - 2

            # phonon part
            for iph in range(mol_list[imol].nphs):
                nqb = mol_list[imol].phs[iph].nqboson
                if nqb == 1:
                    mpo = np.zeros([MPOdim[impo], pbond_list[impo], pbond_list[impo], MPOdim[impo + 1]])
                    for ibra in range(pbond_list[impo]):
                        for iket in range(pbond_list[impo]):
                            # first column
                            mpo[0, ibra, iket, 0] = PhElementOpera("Iden", ibra, iket)
                            mpo[-1, ibra, iket, 0] = PhElementOpera("b^\dagger b",
                                                                    ibra, iket) * mol_list[imol].phs[iph].omega[0] \
                                                     + PhElementOpera("(b^\dagger + b)^3", ibra, iket) * \
                                                       mol_list[imol].phs[iph].force3rd[0] * (0.5 / mol_list[imol].phs[iph].omega[
                                0]) ** 1.5
                            if rep == "chain" and iph != 0:
                                mpo[1, ibra, iket, 0] = PhElementOpera("b", ibra, iket) * \
                                                        mol_list[imol].phhop[iph, iph - 1]
                                mpo[2, ibra, iket, 0] = PhElementOpera("b^\dagger", ibra, iket) * \
                                                        mol_list[imol].phhop[iph, iph - 1]
                            else:
                                mpo[1, ibra, iket, 0] = PhElementOpera("b^\dagger + b", ibra, iket) * \
                                                        (mol_list[imol].phs[iph].omega[1] ** 2 /
                                                         np.sqrt(2. * mol_list[imol].phs[iph].omega[0]) * -
                                                         mol_list[imol].phs[iph].dis[1]
                                                         + 3.0 * mol_list[imol].phs[iph].dis[1] ** 2 *
                                                         mol_list[imol].phs[iph].force3rd[1] /
                                                         np.sqrt(2. * mol_list[imol].phs[iph].omega[0])) \
                                                        + PhElementOpera("(b^\dagger + b)^2", ibra, iket) * \
                                                          (0.25 * (
                                                              mol_list[imol].phs[iph].omega[1] ** 2 - mol_list[imol].phs[iph].omega[
                                                                  0] ** 2) / mol_list[imol].phs[iph].omega[0]
                                                           - 1.5 * mol_list[imol].phs[iph].dis[1] *
                                                           mol_list[imol].phs[iph].force3rd[1] / mol_list[imol].phs[iph].omega[0]) \
                                                        + PhElementOpera("(b^\dagger + b)^3", ibra, iket) * \
                                                          (mol_list[imol].phs[iph].force3rd[1] - mol_list[imol].phs[iph].force3rd[
                                                              0]) * (0.5 / mol_list[imol].phs[iph].omega[0]) ** 1.5

                            if imol != nmols - 1 or iph != mol_list[imol].nphs - 1:
                                mpo[-1, ibra, iket, -1] = PhElementOpera("Iden", ibra, iket)

                                if rep == "chain":
                                    if iph == 0:
                                        mpo[-1, ibra, iket, 1] = PhElementOpera("b^\dagger", ibra, iket)
                                        mpo[-1, ibra, iket, 2] = PhElementOpera("b", ibra, iket)
                                        for icol in range(3, MPOdim[impo + 1] - 1):
                                            mpo[icol - 1, ibra, iket, icol] = PhElementOpera("Iden", ibra, iket)
                                    elif iph == mol_list[imol].nphs - 1:
                                        for icol in range(1, MPOdim[impo + 1] - 1):
                                            mpo[icol + 2, ibra, iket, icol] = PhElementOpera("Iden", ibra, iket)
                                    else:
                                        mpo[-1, ibra, iket, 1] = PhElementOpera("b^\dagger", ibra, iket)
                                        mpo[-1, ibra, iket, 2] = PhElementOpera("b", ibra, iket)
                                        for icol in range(3, MPOdim[impo + 1] - 1):
                                            mpo[icol, ibra, iket, icol] = PhElementOpera("Iden", ibra, iket)

                                elif rep == "star":
                                    if iph != mol_list[imol].nphs - 1:
                                        for icol in range(1, MPOdim[impo + 1] - 1):
                                            mpo[icol, ibra, iket, icol] = PhElementOpera("Iden", ibra, iket)
                                    else:
                                        for icol in range(1, MPOdim[impo + 1] - 1):
                                            mpo[icol + 1, ibra, iket, icol] = PhElementOpera("Iden", ibra, iket)
                    self.append(mpo)
                    impo += 1
                else:
                    # b + b^\dagger in Mpo representation
                    for iqb in range(nqb):
                        mpo = np.zeros([MPOdim[impo], pbond_list[impo], pbond_list[impo], MPOdim[impo + 1]])

                        if rep == "star":
                            bpbdagger = qbopera[imol]["bpbdagger" + str(iph)][iqb]

                            for ibra in range(mol_list[imol].phs[iph].base):
                                for iket in range(mol_list[imol].phs[iph].base):
                                    mpo[0, ibra, iket, 0] = PhElementOpera("Iden", ibra, iket)
                                    mpo[-1, ibra, iket, 0] = PhElementOpera("b^\dagger b",
                                                                            ibra, iket) * mol_list[imol].phs[iph].omega[0] * \
                                                             float(mol_list[imol].phs[iph].base) ** (nqb - iqb - 1)

                                    #  the # of identity operator 
                                    if iqb != nqb - 1:
                                        nI = MPOdim[impo + 1] - bpbdagger.shape[-1] - 1
                                    else:
                                        nI = MPOdim[impo + 1] - 1

                                    for iset in range(1, nI + 1):
                                        mpo[-iset, ibra, iket, -iset] = PhElementOpera("Iden", ibra, iket)

                            # b + b^\dagger 
                            if iqb != nqb - 1:
                                mpo[1:bpbdagger.shape[0] + 1, :, :, 1:bpbdagger.shape[-1] + 1] = bpbdagger
                            else:
                                mpo[1:bpbdagger.shape[0] + 1, :, :, 0:bpbdagger.shape[-1]] = bpbdagger

                        elif rep == "chain":

                            b = qbopera[imol]["b" + str(iph)][iqb]
                            bdagger = qbopera[imol]["bdagger" + str(iph)][iqb]
                            bpbdagger = qbopera[imol]["bpbdagger" + str(iph)][iqb]

                            for ibra in range(mol_list[imol].phs[iph].base):
                                for iket in range(mol_list[imol].phs[iph].base):
                                    mpo[0, ibra, iket, 0] = PhElementOpera("Iden", ibra, iket)
                                    mpo[-1, ibra, iket, 0] = PhElementOpera("b^\dagger b",
                                                                            ibra, iket) * mol_list[imol].phs[iph].omega[0] * \
                                                             float(mol_list[imol].phs[iph].base) ** (nqb - iqb - 1)

                                    #  the # of identity operator 
                                    if impo == len(MPOdim) - 2:
                                        nI = nIe - 1
                                    else:
                                        nI = nIe

                                    # print
                                    # "nI", nI
                                    for iset in range(1, nI + 1):
                                        mpo[-iset, ibra, iket, -iset] = PhElementOpera("Iden", ibra, iket)

                            if iph == 0:
                                # b + b^\dagger 
                                if iqb != nqb - 1:
                                    mpo[1:bpbdagger.shape[0] + 1, :, :, 1:bpbdagger.shape[-1] + 1] = bpbdagger
                                else:
                                    mpo[1:bpbdagger.shape[0] + 1, :, :, 0:1] = \
                                        bpbdagger * \
                                        mol_list[imol].phs[iph].omega[1] ** 2 / np.sqrt(2. * mol_list[imol].phs[iph].omega[0]) \
                                        * -mol_list[imol].phs[iph].dis[1]
                            else:
                                # b^\dagger, b
                                if iqb != nqb - 1:
                                    mpo[1:b.shape[0] + 1, :, :, 1:b.shape[-1] + 1] = b
                                    mpo[b.shape[0] + 1:b.shape[0] + 1 + bdagger.shape[0], :, :, \
                                    b.shape[-1] + 1:b.shape[-1] + 1 + bdagger.shape[-1]] = bdagger
                                else:
                                    mpo[1:b.shape[0] + 1, :, :, 0:1] = b * mol_list[imol].phhop[iph, iph - 1]
                                    mpo[b.shape[0] + 1:b.shape[0] + 1 + bdagger.shape[0], :, :, 0:1] \
                                        = bdagger * mol_list[imol].phhop[iph, iph - 1]

                            if iph != mol_list[imol].nphs - 1:
                                if iph == 0:
                                    loffset = bpbdagger.shape[0]
                                    roffset = bpbdagger.shape[-1]
                                else:
                                    loffset = b.shape[0] + bdagger.shape[0]
                                    roffset = b.shape[-1] + bdagger.shape[-1]
                                    # b^\dagger, b     
                                if iqb == 0:
                                    mpo[-1:, :, :, roffset + 1:roffset + 1 + bdagger.shape[-1]] = bdagger
                                    mpo[-1:, :, :,
                                    roffset + 1 + bdagger.shape[-1]:roffset + 1 + bdagger.shape[-1] + b.shape[-1]] = b
                                elif iqb == nqb - 1:
                                    # print
                                    # "He", loffset + 1, \
                                    # loffset + 1 + bdagger.shape[0], loffset + 1 + bdagger.shape[0] + b.shape[0],
                                    mpo[loffset + 1:loffset + 1 + bdagger.shape[0], :, :, 1:2] = bdagger
                                    mpo[loffset + 1 + bdagger.shape[0]:loffset + 1 + bdagger.shape[0] + b.shape[0], :,
                                    :, 2:3] = b
                                else:
                                    mpo[loffset + 1:loffset + 1 + bdagger.shape[0], :, :, \
                                    roffset + 1:roffset + 1 + bdagger.shape[-1]] = bdagger
                                    mpo[loffset + 1 + bdagger.shape[0]:loffset + 1 + bdagger.shape[0] + b.shape[0], :,
                                    :, \
                                    roffset + 1 + bdagger.shape[-1]:roffset + 1 + bdagger.shape[-1] + b.shape[-1]] = b

                        self.append(mpo)
                        impo += 1

        self.pbond_list = pbond_list
        self.mpo_dim = MPOdim
        self.mpo_qn_idx = MPOQNidx
        self.mpo_qn_tot = MPOQNtot
