from __future__ import absolute_import, print_function, unicode_literals

import numpy as np
import scipy

from ephMPS.mps.matrix import MatrixOp
from ephMPS.mps.mp import MatrixProduct
from ephMPS.mps.mps import Mps
from ephMPS.mps.ephtable import EphTable
from ephMPS.mps.elementop import EElementOpera, PhElementOpera
from ephMPS.utils.utils import roundrobin


def base_convert(n, base):
    """
    convert 10 base number to any base number
    """
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
    def quasi_boson(cls, opera, nqb, trunc, base=2, C1=1.0, C2=1.0):
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
                    lstring = np.array(map(int, base_convert(i, base).zfill(nqb)))
                    rstring = np.array(map(int, base_convert(i - 1, base).zfill(nqb)))
                    pos = tuple(roundrobin(lstring, rstring))
                    mat[pos] = np.sqrt(i)

            if opera == "b + b^\dagger" or opera == "b":
                for i in range(0, base ** nqb - 1):
                    # b
                    lstring = np.array(map(int, base_convert(i, base).zfill(nqb)))
                    rstring = np.array(map(int, base_convert(i + 1, base).zfill(nqb)))
                    pos = tuple(roundrobin(lstring, rstring))
                    mat[pos] = np.sqrt(i + 1)

        elif opera == "C1(b + b^\dagger) + C2(b + b^\dagger)^2":
            # b^+
            for i in range(1, base ** nqb):
                lstring = np.array(map(int, base_convert(i, base).zfill(nqb)))
                rstring = np.array(map(int, base_convert(i - 1, base).zfill(nqb)))
                pos = tuple(roundrobin(lstring, rstring))
                mat[pos] = C1 * np.sqrt(i)
            # b
            for i in range(0, base ** nqb - 1):
                lstring = np.array(map(int, base_convert(i, base).zfill(nqb)))
                rstring = np.array(map(int, base_convert(i + 1, base).zfill(nqb)))
                pos = tuple(roundrobin(lstring, rstring))
                mat[pos] = C1 * np.sqrt(i + 1)
            # bb
            for i in range(0, base ** nqb - 2):
                lstring = np.array(map(int, base_convert(i, base).zfill(nqb)))
                rstring = np.array(map(int, base_convert(i + 2, base).zfill(nqb)))
                pos = tuple(roundrobin(lstring, rstring))
                mat[pos] = C2 * np.sqrt(i + 2) * np.sqrt(i + 1)
            # b^\dagger b^\dagger
            for i in range(2, base ** nqb):
                lstring = np.array(map(int, base_convert(i, base).zfill(nqb)))
                rstring = np.array(map(int, base_convert(i - 2, base).zfill(nqb)))
                pos = tuple(roundrobin(lstring, rstring))
                mat[pos] = C2 * np.sqrt(i) * np.sqrt(i - 1)
            # b^\dagger b + b b^\dagger
            for i in range(0, base ** nqb):
                lstring = np.array(map(int, base_convert(i, base).zfill(nqb)))
                rstring = np.array(map(int, base_convert(i, base).zfill(nqb)))
                pos = tuple(roundrobin(lstring, rstring))
                mat[pos] = C2 * float(i * 2 + 1)

        elif opera == "b^\dagger b":
            # actually Identity operator can be constructed directly
            for i in range(0, base ** nqb):
                # I
                lstring = np.array(map(int, base_convert(i, base).zfill(nqb)))
                rstring = np.array(map(int, base_convert(i, base).zfill(nqb)))
                pos = tuple(roundrobin(lstring, rstring))
                mat[pos] = float(i)

        # check the original mat
        # mat = np.moveaxis(mat,range(1,nqb*2,2),range(nqb,nqb*2))
        # print mat.reshape(base**nqb,base**nqb)

        # decompose canonicalise
        MPO = cls()
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
        MPOnew = MPO.compress(thresh=trunc)
        # print "trunc", trunc, "distance", mpslib.distance(MPO,MPOnew)
        # fidelity = mpslib.dot(mpslib.conj(MPOnew), MPO) / mpslib.dot(mpslib.conj(MPO), MPO)
        # print "compression fidelity:: ", fidelity
        # print "compressed MPO shape", [i.shape[0] for i in MPOnew] + [1]

        return MPOnew

    @classmethod
    def from_mps(cls, mps):
        mpo = cls()
        for ms in mps:
            mo = np.zeros([ms.shape[0]]+[ms.shape[1]]*2+[ms.shape[2]])
            for iaxis in range(ms.shape[1]):
                mo[:, iaxis, iaxis, :] = ms[:, iaxis, :].copy()
            mpo.append(mo)
        mpo.qn = mps.qn.copy()
        mpo.qntot = mps.qntot
        mpo.qnidx = mps.qnidx
        mpo.thresh = mps.thresh
        return mpo

    @classmethod
    def exact_propagator(cls, mol_list, x, space="GS", shift=0.0):
        '''
        construct the GS space propagator e^{xH} exact MPO
        H=\sum_{in} \omega_{in} b^\dagger_{in} b_{in}
        fortunately, the H is local. so e^{xH} = e^{xh1}e^{xh2}...e^{xhn}
        the bond dimension is 1
        shift is the a constant for H+shift
        '''
        assert space in ["GS", "EX"]

        mpo = cls()


        for imol, mol in enumerate(mol_list):
            e_pbond = mol.pbond[0]
            mo = np.zeros([1, e_pbond, e_pbond, 1])
            for ibra in range(e_pbond):
                mo[0, ibra, ibra, 0] = 1.0
            mpo.append(mo)

            for iph, ph in enumerate(mol.phs):

                if space == "EX":
                    # for the EX space, with quasiboson algorithm, the b^\dagger + b
                    # operator is not local anymore.
                    assert ph.nqboson == 1
                    ph_pbond = ph.pbond[0]
                    # construct the matrix exponential by diagonalize the matrix first
                    h_mo = np.zeros([ph_pbond, ph_pbond])

                    for ibra in range(ph_pbond):
                        for iket in range(ph_pbond):
                            h_mo[ibra, iket] = PhElementOpera("b^\dagger b", ibra, iket) * ph.omega[0] \
                                               + PhElementOpera("(b^\dagger + b)^3", ibra, iket) * \
                                                 ph.force3rd[0] * (0.5 / ph.omega[0]) ** 1.5 \
                                               + PhElementOpera("b^\dagger + b", ibra, iket) * \
                                                 (ph.omega[1] ** 2 / np.sqrt(2. * ph.omega[0]) * - ph.dis[1] + 3.0 *
                                                  ph.dis[1] ** 2 * ph.force3rd[1] / np.sqrt(2. * ph.omega[0])) \
                                               + PhElementOpera("(b^\dagger + b)^2", ibra, iket) * \
                                                 (0.25 * (ph.omega[1] ** 2 - ph.omega[0] ** 2) / ph.omega[0] - 1.5 *
                                                  ph.dis[1] * ph.force3rd[1] / ph.omega[0]) \
                                               + PhElementOpera("(b^\dagger + b)^3", ibra, iket) * \
                                                 (ph.force3rd[1] - ph.force3rd[0]) * (0.5 / ph.omega[0]) ** 1.5

                    w, v = scipy.linalg.eigh(h_mo)
                    h_mo = np.diag(np.exp(x * w))
                    h_mo = v.dot(h_mo)
                    h_mo = h_mo.dot(v.T)

                    mo = np.zeros([1, ph_pbond, ph_pbond, 1], dtype=np.complex128)
                    mo[0, :, :, 0] = h_mo

                    mpo.append(mo)

                elif space == "GS":
                    # for the ground state space, yet doesn't support 3rd force
                    # potential quasiboson algorithm
                    ph_pbond = ph.pbond[0]
                    for i in ph.force3rd:
                        anharmo = not np.allclose(ph.force3rd[i] * ph.dis[i] / ph.omega[i], 0.0)
                        if anharmo:
                            break
                    if not anharmo:
                        for iboson in range(ph.nqboson):
                            mo = np.zeros([1, ph_pbond, ph_pbond, 1],
                                          dtype=np.complex128)

                            for ibra in range(ph_pbond):
                                mo[0, ibra, ibra, 0] = np.exp(
                                    x * ph.omega[0] * float(ph.base) ** (ph.nqboson - iboson - 1) * float(ibra))

                            mpo.append(mo)
                    else:
                        assert ph.nqboson == 1
                        # construct the matrix exponential by diagonalize the matrix first
                        h_mo = np.zeros([ph_pbond, ph_pbond])
                        for ibra in range(ph_pbond):
                            for iket in range(ph_pbond):
                                h_mo[ibra, iket] = PhElementOpera("b^\dagger b", ibra, iket) * \
                                                   ph.omega[0] \
                                                   + PhElementOpera("(b^\dagger + b)^3", ibra, iket) * \
                                                     ph.force3rd[0] * (0.5 / ph.omega[0]) ** 1.5

                        w, v = scipy.linalg.eigh(h_mo)
                        h_mo = np.diag(np.exp(x * w))
                        h_mo = v.dot(h_mo)
                        h_mo = h_mo.dot(v.T)

                        mo = np.zeros([1, ph_pbond, ph_pbond, 1],
                                      dtype=np.complex128)
                        mo[0, :, :, 0] = h_mo

                        mpo.append(mo)

        # shift the H by plus a constant

        mpo = mpo.scale(np.exp(shift * x))

        mpo.qn = [[0]] * (len(mpo) + 1)
        mpo.qnidx = len(mpo) - 1
        mpo.qntot = 0


        return mpo

    @classmethod
    def approx_propagator(cls, mpo, dt, prop_method='C_RK4', thresh=0, compress_method="svd"):
        '''
        e^-iHdt : approximate propagator MPO from Runge-Kutta methods
        '''

        mps = Mps()
        mps.dim = [1] * (mpo.site_num + 1)
        mps.qn = [[0]] * (mpo.site_num + 1)
        mps.qnidx = mpo.site_num - 1
        mps.qntot = 0
        mps.thresh = thresh

        for impo in range(mpo.site_num):
            ms = np.ones([1, mpo[impo].shape[1], 1], dtype=np.complex128)
            mps.append(ms)
        approx_mpo_t0 = Mpo.from_mps(mps)

        approx_mpo = approx_mpo_t0.evolve(mpo, dt, prop_method=prop_method, thresh=thresh,
                                          compress_method=compress_method)

        #print"approx propagator thresh:", thresh
        #if QNargs is not None:
            #print "approx propagator dim:", [mpo.shape[0] for mpo in approxMPO[0]]
        #else:
            # print "approx propagator dim:", [mpo.shape[0] for mpo in approxMPO]

        #chkIden = mpslib.mapply(mpslib.conj(approxMPO, QNargs=QNargs), approxMPO, QNargs=QNargs)
        #print "approx propagator Identity error", np.sqrt(mpslib.distance(chkIden, IMPO, QNargs=QNargs) / \
        #                                            mpslib.dot(IMPO, IMPO, QNargs=QNargs))

        return approx_mpo


    @classmethod
    def onsite(cls, mol_list, pbond, opera, dipole=False):
        assert opera in ["a", "a^\dagger", "a^\dagger a"]
        nmols = len(mol_list)

        MPOdim = []
        for imol in range(nmols):
            MPOdim.append(2)
            for ph in mol_list[imol].phs:
                for iboson in range(ph.nqboson):
                    if imol != nmols - 1:
                        MPOdim.append(2)
                    else:
                        MPOdim.append(1)

        MPOdim[0] = 1
        MPOdim.append(1)
        #print opera, "operator MPOdim", MPOdim

        mpo = cls()
        impo = 0
        for imol in range(nmols):
            mo = np.zeros([MPOdim[impo], pbond[impo], pbond[impo], MPOdim[impo + 1]])
            for ibra in range(pbond[impo]):
                for iket in range(pbond[impo]):
                    if not dipole:
                        mo[-1, ibra, iket, 0] = EElementOpera(opera, ibra, iket)
                    else:
                        mo[-1, ibra, iket, 0] = EElementOpera(opera, ibra, iket) * mol_list[imol].dipole
                    if imol != 0:
                        mo[0, ibra, iket, 0] = EElementOpera("Iden", ibra, iket)
                    if imol != nmols - 1:
                        mo[-1, ibra, iket, -1] = EElementOpera("Iden", ibra, iket)
            mpo.append(mo)
            impo += 1

            for ph in mol_list[imol].phs:
                for iboson in range(ph.nqboson):
                    mo = np.zeros([MPOdim[impo], pbond[impo], pbond[impo], MPOdim[impo + 1]])
                    for ibra in range(pbond[impo]):
                        for idiag in range(MPOdim[impo]):
                            mo[idiag, ibra, ibra, idiag] = 1.0

                    mpo.append(mo)
                    impo += 1

        # quantum number part
        # len(MPO)-1 = len(MPOQN)-2, the L-most site is R-qn
        mpo.qnidx = len(mpo) - 1

        totnqboson = 0
        for ph in mol_list[-1].phs:
            totnqboson += ph.nqboson

        if opera == "a":
            mpo.qn = [[0]] + [[-1, 0]] * (len(mpo) - totnqboson - 1) + [[-1]] * (totnqboson + 1)
            mpo.qntot = -1
        elif opera == "a^\dagger":
            mpo.qn = [[0]] + [[1, 0]] * (len(mpo) - totnqboson - 1) + [[1]] * (totnqboson + 1)
            mpo.qntot = 1
        elif opera == "a^\dagger a":
            mpo.qn = [[0]] + [[0, 0]] * (len(mpo) - totnqboson - 1) + [[0]] * (totnqboson + 1)
            mpo.qntot = 0
        mpo.qn[-1] = [0]

        return mpo


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
                for iph in range(mol.nphs):
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
                for iph in range(mol.nphs):
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
                for iph in range(mol.nphs):
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
            for iph in range(mol.nphs):
                nqb = mol.phs[iph].nqboson
                if nqb != 1:
                    if rep == "chain":
                        b = Mpo.quasi_boson("b", nqb,
                                            mol.phs[iph].qbtrunc, base=mol.phs[iph].base)
                        bdagger = Mpo.quasi_boson("b^\dagger", nqb,
                                                  mol.phs[iph].qbtrunc, base=mol.phs[iph].base)
                        bpbdagger = Mpo.quasi_boson("b + b^\dagger", nqb,
                                                    mol.phs[iph].qbtrunc, base=mol.phs[iph].base)
                        qbopera[imol]["b" + str(iph)] = b
                        qbopera[imol]["bdagger" + str(iph)] = bdagger
                        qbopera[imol]["bpbdagger" + str(iph)] = bpbdagger

                        if iph == 0:
                            if iph != mol.nphs - 1:
                                addmpodim = [b[i].shape[0] + bdagger[i].shape[0] + bpbdagger[i].shape[0] - 1 for i in
                                             range(nqb)]
                            else:
                                addmpodim = [bpbdagger[i].shape[0] - 1 for i in range(nqb)]
                            addmpodim[0] = 0
                        else:
                            addmpodim = [(b[i].shape[0] + bdagger[i].shape[0]) * 2 - 2 for i in range((nqb))]
                            addmpodim[0] = 0

                    else:
                        bpbdagger = Mpo.quasi_boson("C1(b + b^\dagger) + C2(b + b^\dagger)^2", nqb,
                                                    mol.phs[iph].qbtrunc,
                                                    base=mol.phs[iph].base,
                                                    C1=mol.phs[iph].omega[1] ** 2 / np.sqrt(
                                                        2. * mol.phs[iph].omega[0]) * -
                                                       mol.phs[iph].dis[1],
                                                    C2=0.25 * (mol.phs[iph].omega[1] ** 2 -
                                                               mol.phs[iph].omega[0] ** 2) /
                                                       mol.phs[iph].omega[0])

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
                    elocal = mol.elocalex
                    if elocal_offset is not None:
                        elocal += elocal_offset[imol]
                    mpo[-1, ibra, iket, 0] = EElementOpera("a^\dagger a", ibra, iket) \
                                             * (elocal + mol.e0)
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
                                mpo[-1, ibra, iket, (nmols - jmol) * 2] = EElementOpera("a^\dagger", ibra, iket) * \
                                                                          J_matrix[
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
                                                                                                           iket) * \
                                                                                             J_matrix[
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
            for iph in range(mol.nphs):
                nqb = mol.phs[iph].nqboson
                if nqb == 1:
                    mpo = np.zeros([MPOdim[impo], pbond_list[impo], pbond_list[impo], MPOdim[impo + 1]])
                    for ibra in range(pbond_list[impo]):
                        for iket in range(pbond_list[impo]):
                            # first column
                            mpo[0, ibra, iket, 0] = PhElementOpera("Iden", ibra, iket)
                            mpo[-1, ibra, iket, 0] = PhElementOpera("b^\dagger b",
                                                                    ibra, iket) * mol.phs[iph].omega[0] \
                                                     + PhElementOpera("(b^\dagger + b)^3", ibra, iket) * \
                                                       mol.phs[iph].force3rd[0] * (0.5 / mol.phs[
                                iph].omega[
                                0]) ** 1.5
                            if rep == "chain" and iph != 0:
                                mpo[1, ibra, iket, 0] = PhElementOpera("b", ibra, iket) * \
                                                        mol.phhop[iph, iph - 1]
                                mpo[2, ibra, iket, 0] = PhElementOpera("b^\dagger", ibra, iket) * \
                                                        mol.phhop[iph, iph - 1]
                            else:
                                mpo[1, ibra, iket, 0] = PhElementOpera("b^\dagger + b", ibra, iket) * \
                                                        (mol.phs[iph].omega[1] ** 2 /
                                                         np.sqrt(2. * mol.phs[iph].omega[0]) * -
                                                         mol.phs[iph].dis[1]
                                                         + 3.0 * mol.phs[iph].dis[1] ** 2 *
                                                         mol.phs[iph].force3rd[1] /
                                                         np.sqrt(2. * mol.phs[iph].omega[0])) \
                                                        + PhElementOpera("(b^\dagger + b)^2", ibra, iket) * \
                                                          (0.25 * (
                                                              mol.phs[iph].omega[1] ** 2 -
                                                              mol.phs[iph].omega[
                                                                  0] ** 2) / mol.phs[iph].omega[0]
                                                           - 1.5 * mol.phs[iph].dis[1] *
                                                           mol.phs[iph].force3rd[1] /
                                                           mol.phs[iph].omega[0]) \
                                                        + PhElementOpera("(b^\dagger + b)^3", ibra, iket) * \
                                                          (mol.phs[iph].force3rd[1] -
                                                           mol.phs[iph].force3rd[
                                                               0]) * (0.5 / mol.phs[iph].omega[0]) ** 1.5

                            if imol != nmols - 1 or iph != mol.nphs - 1:
                                mpo[-1, ibra, iket, -1] = PhElementOpera("Iden", ibra, iket)

                                if rep == "chain":
                                    if iph == 0:
                                        mpo[-1, ibra, iket, 1] = PhElementOpera("b^\dagger", ibra, iket)
                                        mpo[-1, ibra, iket, 2] = PhElementOpera("b", ibra, iket)
                                        for icol in range(3, MPOdim[impo + 1] - 1):
                                            mpo[icol - 1, ibra, iket, icol] = PhElementOpera("Iden", ibra, iket)
                                    elif iph == mol.nphs - 1:
                                        for icol in range(1, MPOdim[impo + 1] - 1):
                                            mpo[icol + 2, ibra, iket, icol] = PhElementOpera("Iden", ibra, iket)
                                    else:
                                        mpo[-1, ibra, iket, 1] = PhElementOpera("b^\dagger", ibra, iket)
                                        mpo[-1, ibra, iket, 2] = PhElementOpera("b", ibra, iket)
                                        for icol in range(3, MPOdim[impo + 1] - 1):
                                            mpo[icol, ibra, iket, icol] = PhElementOpera("Iden", ibra, iket)

                                elif rep == "star":
                                    if iph != mol.nphs - 1:
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

                            for ibra in range(mol.phs[iph].base):
                                for iket in range(mol.phs[iph].base):
                                    mpo[0, ibra, iket, 0] = PhElementOpera("Iden", ibra, iket)
                                    mpo[-1, ibra, iket, 0] = PhElementOpera("b^\dagger b",
                                                                            ibra, iket) * mol.phs[iph].omega[
                                                                 0] * \
                                                             float(mol.phs[iph].base) ** (nqb - iqb - 1)

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

                            for ibra in range(mol.phs[iph].base):
                                for iket in range(mol.phs[iph].base):
                                    mpo[0, ibra, iket, 0] = PhElementOpera("Iden", ibra, iket)
                                    mpo[-1, ibra, iket, 0] = PhElementOpera("b^\dagger b",
                                                                            ibra, iket) * mol.phs[iph].omega[
                                                                 0] * \
                                                             float(mol.phs[iph].base) ** (nqb - iqb - 1)

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
                                        mol.phs[iph].omega[1] ** 2 / np.sqrt(
                                            2. * mol.phs[iph].omega[0]) \
                                        * -mol.phs[iph].dis[1]
                            else:
                                # b^\dagger, b
                                if iqb != nqb - 1:
                                    mpo[1:b.shape[0] + 1, :, :, 1:b.shape[-1] + 1] = b
                                    mpo[b.shape[0] + 1:b.shape[0] + 1 + bdagger.shape[0], :, :, \
                                    b.shape[-1] + 1:b.shape[-1] + 1 + bdagger.shape[-1]] = bdagger
                                else:
                                    mpo[1:b.shape[0] + 1, :, :, 0:1] = b * mol.phhop[iph, iph - 1]
                                    mpo[b.shape[0] + 1:b.shape[0] + 1 + bdagger.shape[0], :, :, 0:1] \
                                        = bdagger * mol.phhop[iph, iph - 1]

                            if iph != mol.nphs - 1:
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
        self.dim = MPOdim
        self.qn = MPOQN
        self.qnidx = MPOQNidx
        self.qntot = MPOQNtot

    def apply(self, mp):
        new_mps = mp.copy()
        if mp.is_mps:
            # mpo x mps
            for i, (mt_self, mt_other) in enumerate(zip(self, mp)):
                assert mt_self.shape[2] == mt_other.shape[1]
                # mt=np.einsum("apqb,cqd->acpbd",mpo[i],mps[i])
                mt = np.moveaxis(np.tensordot(mt_self, mt_other, axes=([2], [1])), 3, 1)
                mt = np.reshape(mt, [mt_self.shape[0] * mt_other.shape[0], mt_self.shape[1],
                                     mt_self.shape[-1] * mt_other.shape[-1]])
                new_mps[i] = mt
        elif mp.is_mpo:
            # mpo x mpo
            for i, (mt_self, mt_other) in enumerate(zip(self, mp)):
                assert mt_self.shape[2] == mt_other.shape[1]
                # mt=np.einsum("apqb,cqrd->acprbd",mt_o,mt_s)
                mt = np.moveaxis(np.tensordot(mt_self, mt_other, axes=([2], [1])), [-3, -2], [1, 3])
                mt = np.reshape(mt, [mt_self.shape[0] * mt_other.shape[0],
                                     mt_self.shape[1], mt_other.shape[2],
                                     mt_self.shape[-1] * mt_other.shape[-1]])
                new_mps[i] = mt
        else:
            assert False
        if self.enable_qn and mp.enable_qn:
            orig_idx = new_mps.qnidx
            new_mps.move_qnidx(self.qnidx)
            new_mps.qn = [np.add.outer(np.array(qn_o), np.array(qn_m)).ravel().tolist()
                          for qn_o, qn_m in zip(self.qn, new_mps.qn)]
            new_mps.move_qnidx(orig_idx)
            new_mps.qntot += self.qntot
        return new_mps

    def contract(self, mps, ncanonical=1, compress_method='svd'):
        assert compress_method in ["svd", "variational"]
        if compress_method == 'svd':
            return self.contract_svd(mps, ncanonical)
        else:
            return self.contract_variational()

    def contract_svd(self, mps, ncanonical=1):

        """
        mapply->canonicalise->compress
        """
        new_mps = self.apply(mps)
        # roundoff can cause problems,
        # so do multiple canonicalisations
        for i in range(ncanonical):
            new_mps.canonicalise()
        new_mps.compress()
        return new_mps

    def contract_variational(self):
        raise NotImplementedError
