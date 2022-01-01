# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

from functools import reduce
from collections import deque

from renormalizer.mps.backend import np, backend, xp
from renormalizer.mps.matrix import (Matrix, multi_tensor_contract, asxp,
    asnumpy, tensordot)


class Environ:
    def __init__(self, mps, mpo, domain=None, mps_conj=None):
        # todo: real disk and other backend
        # todo: contract_one_site_multi_mpo could generalize contract_one_site,
        # we could unify them in the future.

        # idx indicates the exact position of L or R, like
        # L(idx-1) - mpo(idx) - R(idx+1)
        self._virtual_disk = {}
        if type(mpo) is list:
            ndim = len(mpo) + 2
        else:
            ndim = 3
        self.sentinel = xp.ones([1,]*ndim, dtype=backend.real_dtype)
        self._construct(mps, mpo, domain, mps_conj)

    def _construct(self, mps, mpo, domain=None, mps_conj=None):

        assert domain in ["L", "R", None]

        if mps_conj is None:
            mps_conj = mps.conj()

        if domain is None:
            self._construct(mps, mpo, "L", mps_conj)
            self._construct(mps, mpo, "R", mps_conj)
            return
        if domain == "L":
            start, end, inc = 0, len(mps) - 1, 1
        else:
            start, end, inc = len(mps) - 1, 0, -1
        self.write_l_sentinel(mps)
        self.write_r_sentinel(mps)

        tensor = self.sentinel
        for idx in range(start, end, inc):
            if type(mpo) is list:
                # a list of mpos
                tensor = contract_one_site_multi_mpo(tensor, mps[idx], [mp[idx] for mp in mpo], domain, ms_conj=mps_conj[idx])
            else:
                # one single mpo
                tensor = contract_one_site(tensor, mps[idx], mpo[idx], domain, ms_conj=mps_conj[idx])
            self.write(domain, idx, tensor)

    def write_l_sentinel(self, mps):
        self.write("L", -1, self.sentinel)

    def write_r_sentinel(self, mps):
        self.write("R", len(mps), self.sentinel)

    def GetLR(
        self, domain, siteidx, mps, mpo, itensor=None, method="Scratch", mps_conj=None):
        """
        get the L/R Hamiltonian matrix at a random site(siteidx): 3d tensor
        S-     -S     mpsconj
        O- or  -O     mpo
        S-     -S     mps
        enviroment part from self.virtual_disk,  system part from one step calculation
        support from scratch calculation: from two open boundary np.ones((1,1,1))
        """

        assert domain in ["L", "R"]
        assert method in ["Enviro", "System", "Scratch"]
        if mps_conj is None:
            # provide a dummy value. Creating conjugation of a whole MPS should be avoided when possible
            # since the operation is actually rather expensive.
            mps_conj = [None] * len(mps)

        if siteidx not in range(len(mps)):
            return self.sentinel

        if method == "Scratch":
            itensor = self.sentinel
            if domain == "L":
                sitelist = range(siteidx + 1)
            else:
                sitelist = range(len(mps) - 1, siteidx - 1, -1)
            for imps in sitelist:
                if type(mpo) is list:
                    itensor = contract_one_site_multi_mpo(itensor, mps[imps],
                            [mp[imps] for mp in mpo], domain,
                            ms_conj=mps_conj[imps])
                else:
                    itensor = contract_one_site(itensor, mps[imps], mpo[imps],
                        domain, ms_conj=mps_conj[imps])
        elif method == "Enviro":
            itensor = self.read(domain, siteidx)
        elif method == "System":
            if itensor is None:
                offset = -1 if domain == "L" else 1
                itensor = self.read(domain, siteidx + offset)
            if type(mpo) is list:
                itensor = contract_one_site_multi_mpo(itensor, mps[siteidx],
                        [mp[siteidx] for mp in mpo],
                        domain, mps_conj[siteidx])
            else:
                itensor = contract_one_site(itensor, mps[siteidx], mpo[siteidx],
                        domain, mps_conj[siteidx])
            self.write(domain, siteidx, itensor)

        return itensor

    def write(self, domain, siteidx, tensor):
        self._virtual_disk[(domain, siteidx)] = asnumpy(tensor)

    def read(self, domain: str, siteidx: int):
        return asxp(self._virtual_disk[(domain, siteidx)])


def contract_one_site_multi_mpo(environ, ms, mos, domain, ms_conj=None):
    """
    contract one mpos/mps(mpdm) site, mos is a list containing several local mo
             _   _
            | | | |
    S-S-    | S-|-S-
    O-O- or | O-|-O- (the ancillary bond is traced)
    O-O-    | O-|-O-
    S-S-    | S-|-S-
            |_| |_|
    """
    assert domain in ["L", "R"]
    if ms_conj is None:
        ms_conj = ms.conj()
    if domain == "L":
        if ms.ndim == 3:
            outtensor = tensordot(environ, ms_conj, ([0], [0]))
            for mo in mos:
                outtensor = tensordot(outtensor, mo, ([0,-2], [0,1]))
            outtensor = tensordot(outtensor, ms, ([0,-2], [0,1]))
        elif ms.ndim == 4:
            outtensor = tensordot(environ, ms_conj.transpose(0,2,1,3), ([0], [0]))
            for mo in mos:
                outtensor = tensordot(outtensor, mo, ([0,-2], [0,1]))
            outtensor = tensordot(outtensor, ms, ([0,1,-2], [0,2,1]))
        else:
            raise ValueError(
                f"MPS ndim is not 3 or 4, got {ms.ndim}"
            )
    else:
        if ms.ndim == 3:
            outtensor = tensordot(environ, ms_conj, ([0], [-1]))
            for mo in mos:
                outtensor = tensordot(outtensor, mo, ([0,-1], [-1,1]))
            outtensor = tensordot(outtensor, ms, ([0,-1], [-1,1]))
        elif ms.ndim == 4:
            outtensor = tensordot(environ, ms_conj.transpose(0,2,1,3), ([0], [-1]))
            for mo in mos:
                outtensor = tensordot(outtensor, mo, ([0,-1], [-1,1]))
            outtensor = tensordot(outtensor, ms, ([0,2,-1], [-1,2,1]))
        else:
            raise ValueError(
                f"MPS ndim is not 3 or 4, got {ms.ndim}"
            )

    return outtensor


def contract_one_site(environ, ms, mo, domain, ms_conj=None):
    """
    contract one mpo/mps(mpdm) site
             _   _
            | | | |
    S-S-    | S-|-S-
    O-O- or | O-|-O- (the ancillary bond is traced)
    S-S-    | S-|-S-
            |_| |_|
    """
    assert domain in ["L", "R"]
    if isinstance(ms, Matrix):
        ms = ms.array
    if isinstance(mo, Matrix):
        mo = mo.array
    if ms_conj is None:
        ms_conj = ms.conj()
    if domain == "L":
        assert environ.shape[0] == ms_conj.shape[0]
        assert environ.shape[1] == mo.shape[0]
        assert environ.shape[2] == ms.shape[0]
        """
                       l
        S-a-S-f    O-a-O-f
            d          d
        O-b-O-g or O-b-O-g
            e          e
        S-c-S-h    O-c-O-h
                       l
        """

        if ms.ndim == 3:
            path = [
                ([0, 1], "abc, adf -> bcdf"),
                ([2, 0], "bcdf, bdeg -> cfeg"),
                ([1, 0], "cfeg, ceh -> fgh"),
            ]
        elif ms.ndim == 4:
            path = [
                ([0, 1], "abc, adlf -> bcdlf"),
                ([2, 0], "bcdlf, bdeg -> clfeg"),
                ([1, 0], "clfeg, celh -> fgh"),
            ]
        else:
            raise ValueError(
                f"MPS ndim is not 3 or 4, got {ms.ndim}"
            )
        outtensor = multi_tensor_contract(path, environ, ms_conj, mo, ms)

    else:
        assert environ.shape[0] == ms_conj.shape[-1]
        assert environ.shape[1] == mo.shape[-1]
        assert environ.shape[2] == ms.shape[-1]
        """
                       l
        -f-S-a-S    -f-S-a-S
           d           d
        -g-O-b-O or -g-O-b-O
           e           e
        -h-S-c-S    -h-S-c-S
                       l
        """

        if ms.ndim == 3:
            path = [
                ([0, 1], "fda, abc -> fdbc"),
                ([2, 0], "fdbc, gdeb -> fcge"),
                ([1, 0], "fcge, hec -> fgh"),
            ]
        elif ms.ndim == 4:
            path = [
                ([0, 1], "fdla, abc -> fdlbc"),
                ([2, 0], "fdlbc, gdeb -> flcge"),
                ([1, 0], "flcge, helc -> fgh"),
            ]
        else:
            raise ValueError(
                f"MPS ndim is not 3 or 4, got {ms.ndim}"
            )
        outtensor = multi_tensor_contract(path, ms_conj, environ, mo, ms)

    return outtensor


def select_basis(vset, sset, qnset, compset, Mmax, percent=0):
    """
    select basis to construct new mps, and complementary mps
    vset, compset is the column vector
    """
    # allowed qn subsection
    qnlist = set(qnset)
    # convert to dict
    basdic = dict()
    for i in range(len(qnset)):
        # clean quantum number outside qnlist
        if qnset[i] in qnlist:
            basdic[i] = [qnset[i], sset[i]]

    # each good quantum number block equally get percent/nblocks
    def block_select(basdic, qn, n):
        block_basdic = {i: basdic[i] for i in basdic if basdic[i][0] == qn}
        sort_block_basdic = sorted(
            block_basdic.items(), key=lambda x: x[1][1], reverse=True
        )
        nget = min(n, len(sort_block_basdic))
        # print(qn, "block # of retained basis", nget)
        sidx = [i[0] for i in sort_block_basdic[0:nget]]
        for idx in sidx:
            del basdic[idx]

        return sidx

    nbasis = min(len(basdic), Mmax)
    # print("# of selected basis", nbasis)
    sidx = []

    # equally select from each quantum number block
    if percent != 0:
        nbas_block = int(nbasis * percent / len(qnlist))
        for iqn in qnlist:
            sidx += block_select(basdic, iqn, nbas_block)

    # others
    nbasis = nbasis - len(sidx)

    sortbasdic = sorted(basdic.items(), key=lambda x: x[1][1], reverse=True)
    sidx += [i[0] for i in sortbasdic[0:nbasis]]

    assert len(sidx) == len(set(sidx))  # there must be no duplicated

    mpsdim = len(sidx)
    # need to set value column by column. better in CPU
    ms = np.zeros((vset.shape[0], mpsdim), dtype=vset.dtype)

    if compset is not None:
        compmps = np.zeros((compset.shape[0], mpsdim), dtype=compset.dtype)
    else:
        compmps = None

    mpsqn = []
    stot = 0.0
    for idim in range(mpsdim):
        ms[:, idim] = vset[:, sidx[idim]].copy()
        if (compset is not None) and sidx[idim] < compset.shape[1]:
            compmps[:, idim] = compset[:, sidx[idim]].copy() * sset[sidx[idim]]
        mpsqn.append(qnset[sidx[idim]])
        stot += sset[sidx[idim]] ** 2

    # print("discard:", 1.0 - stot)
    if compmps is not None:
        compmps = asxp(compmps)

    return asxp(ms), mpsdim, mpsqn, compmps


def update_cv(vset, sset, qnset, compset, nexciton, Mmax, spectratype,
              percent=0):
    sidx = select_Xbasis(qnset, sset, range(nexciton + 1), Mmax, spectratype,
                         percent=percent)
    xdim = len(sidx)
    x = np.zeros((vset.shape[0], xdim), dtype=vset.dtype)
    xqn = []
    if compset is not None:
        compx = np.zeros((compset.shape[0], xdim), dtype=compset.dtype)
    else:
        compx = None

    for idim in range(xdim):
        x[:, idim] = vset[:, sidx[idim]].copy()
        if (compset is not None) and (sidx[idim] < compset.shape[1]):
            compx[:, idim] = compset[:, sidx[idim]].copy() * sset[sidx[idim]]
        xqn.append(qnset[sidx[idim]])
    if compx is not None:
        compx = Matrix(compx)
    return Matrix(x), xdim, xqn, compx


def select_Xbasis(qnset, Sset, qnlist, Mmax, spectratype, percent=0.0):
    # select basis according to Sset under qnlist requirement
    # convert to dict
    basdic = {}
    sidx = []
    for i in range(len(qnset)):
        basdic[i] = [qnset[i], Sset[i]]
    # print('basdic', basdic)
    # clean quantum number outiside qnlist
    flag = []
    if spectratype != "conductivity":
        if spectratype == "abs":
            tag_1, tag_2 = 0, 1
        else:
            tag_1, tag_2 = 1, 0
        for ibas in basdic:
            if ((basdic[ibas][0][tag_1] not in qnlist) or (
                    basdic[ibas][0][tag_2] != 0)):
                flag.append(ibas)
    else:
        for ibas in basdic:
            if (basdic[ibas][0][0] not in qnlist) or (
                    basdic[ibas][0][1] not in qnlist):
                flag.append(ibas)

    # i = 0
    # for j in flag:
    #     if i == 0:
    #         del basdic[j]
    #     else:
    #         del basdic[j - i]

    def block_select(basdic, qn, n):
        block_basdic = {i: basdic[i]
                        for i in basdic if basdic[i][0] == qn}
        sort_block_basdic = sorted(block_basdic.items(), key=lambda x: x[1][1],
                                   reverse=True)
        nget = min(n, len(sort_block_basdic))
        # print('n', n)
        # print('len', len(sort_block_basdic))
        # print('nget', nget)
        sidx = [i[0] for i in sort_block_basdic[0: nget]]
        for idx in sidx:
            del basdic[idx]
        # print('qn', qn)
        # print('sidx', sidx)
        return sidx
    nbasis = min(len(basdic), Mmax)
    if percent != 0:
        # print('percent', percent)
        if spectratype == "abs":
            nbas_block = int(nbasis * percent / len(qnlist))
            for iqn in qnlist:
                sidx += block_select(basdic, [iqn, 0], nbas_block)
        elif spectratype == "emi":
            nbas_block = int(nbasis * percent / len(qnlist))
            for iqn in qnlist:
                sidx += block_select(basdic, [0, iqn], nbas_block)
        else:
            nbas_block = int(nbasis * percent / 4)
            for iqn in [[0, 0], [0, 1], [1, 0], [1, 1]]:
                sidx += block_select(basdic, iqn, nbas_block)

    nbasis = nbasis - len(sidx)
    sortbasdic = sorted(basdic.items(), key=lambda y: y[1][1], reverse=True)
    sidx += [i[0] for i in sortbasdic[0: nbasis]]
    # print('sidx', sidx)
    return sidx


def compressed_sum(mps_list, batchsize=5, temp_m_trunc=None):
    assert len(mps_list) != 0
    mps_queue = deque(mps_list)
    while len(mps_queue) != 1:
        term_to_sum = []
        for i in range(min(batchsize, len(mps_queue))):
            term_to_sum.append(mps_queue.popleft())
        s = _sum(term_to_sum, temp_m_trunc=temp_m_trunc)
        mps_queue.append(s)
    return mps_queue[0]


def _sum(mps_list, compress=True, temp_m_trunc=None):
    new_mps = reduce(lambda mps1, mps2: mps1.add(mps2), mps_list)
    if compress and not mps_list[0].compress_add:
        new_mps.canonicalise()
        new_mps.compress(temp_m_trunc=temp_m_trunc)
    return new_mps


def cvec2cmat(cshape, c, qnmat, nexciton, nroots=1):
    # recover good quantum number vector c to matrix format
    if nroots == 1:
        cstruct = np.zeros(cshape, dtype=c.dtype)
        np.place(cstruct, qnmat == nexciton, c)
    else:
        cstruct = []
        if type(c) is not list:
            assert c.ndim == 2
            c = [c[:,iroot] for iroot in range(c.shape[1])]
        for ic in c:
            icstruct = np.zeros(cshape, dtype=ic.dtype)
            np.place(icstruct, qnmat == nexciton, ic)
            cstruct.append(icstruct)

    return cstruct
