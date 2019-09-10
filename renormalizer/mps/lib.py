# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

"""
construct the operator matrix in the MPS sweep procedure
"""
from functools import reduce
from collections import deque

import numpy as np

from renormalizer.mps.matrix import Matrix, multi_tensor_contract, ones, EmptyMatrixError

sentinel = ones((1, 1, 1))


class Environ:
    def __init__(self):
        # todo: real disk and other backend
        # idx indicates the exact position of L or R, like
        # L(idx-1) - mpo(idx) - R(idx+1)
        self.virtual_disk = {}

    def construct(self, mps, mps_conj, mpo, domain):
        tensor = ones((1, 1, 1), mps.dtype)
        assert domain in ["L", "R", "l", "r"]
        if domain == "L" or domain == "l":
            start, end, inc = 0, len(mps) - 1, 1
            self.write(domain, -1, tensor)
        else:
            start, end, inc = len(mps) - 1, 0, -1
            self.write(domain, len(mps), tensor)

        for idx in range(start, end, inc):
            tensor = self.addone(tensor, mps, mps_conj, mpo, idx, domain)
            self.write(domain, idx, tensor)

    def GetLR(
        self, domain, siteidx, MPS, MPSconj, MPO, itensor=sentinel, method="Scratch"
    ):
        """
        get the L/R Hamiltonian matrix at a random site(siteidx): 3d tensor
        S-     -S     MPSconj
        O- or  -O     MPO
        S-     -S     MPS
        enviroment part from self.virtual_disk,  system part from one step calculation
        support from scratch calculation: from two open boundary np.ones((1,1,1))
        """

        assert domain in ["L", "R"]
        assert method in ["Enviro", "System", "Scratch"]

        if siteidx not in range(len(MPS)):
            return sentinel

        if method == "Scratch":
            itensor = sentinel
            if domain == "L":
                sitelist = range(siteidx + 1)
            else:
                sitelist = range(len(MPS) - 1, siteidx - 1, -1)
            for imps in sitelist:
                itensor = self.addone(itensor, MPS, MPSconj, MPO, imps, domain)
        elif method == "Enviro":
            itensor = self.read(domain, siteidx)
        elif method == "System":
            if itensor is None:
                offset = -1 if domain == "L" else 1
                itensor = self.read(domain, siteidx + offset)
            itensor = self.addone(itensor, MPS, MPSconj, MPO, siteidx, domain)
            self.write(domain, siteidx, itensor)

        return itensor

    def addone(self, intensor, MPS, MPSconj, MPO, isite, domain):
        """
        add one MPO/MPS(MPO) site
                 _   _
                | | | |
        S-S-    | S-|-S-
        O-O- or | O-|-O- (the ancillary bond is traced)
        S-S-    | S-|-S-
                |_| |_|
        """
        assert domain in ["L", "R", "l", "r"]

        if domain == "L" or domain == "l":
            assert intensor.shape[0] == MPSconj[isite].shape[0]
            assert intensor.shape[1] == MPO[isite].shape[0]
            assert intensor.shape[2] == MPS[isite].shape[0]
            """
                           l 
            S-a-S-f    O-a-O-f
                d          d
            O-b-O-g or O-b-O-g  
                e          e
            S-c-S-h    O-c-O-h
                           l  
            """

            if MPS[isite].ndim == 3:
                path = [
                    ([0, 1], "abc, adf -> bcdf"),
                    ([2, 0], "bcdf, bdeg -> cfeg"),
                    ([1, 0], "cfeg, ceh -> fgh"),
                ]
            elif MPS[isite].ndim == 4:
                path = [
                    ([0, 1], "abc, adlf -> bcdlf"),
                    ([2, 0], "bcdlf, bdeg -> clfeg"),
                    ([1, 0], "clfeg, celh -> fgh"),
                ]
            else:
                raise ValueError(
                    "MPS ndim at %d is not 3 or 4, got %s" % (isite, MPS[isite].ndim)
                )
            outtensor = multi_tensor_contract(
                path, intensor, MPSconj[isite], MPO[isite], MPS[isite]
            )

        else:
            assert intensor.shape[0] == MPSconj[isite].shape[-1]
            assert intensor.shape[1] == MPO[isite].shape[-1]
            assert intensor.shape[2] == MPS[isite].shape[-1]
            """
                           l
            -f-S-a-S    -f-S-a-S
               d           d
            -g-O-b-O or -g-O-b-O
               e           e
            -h-S-c-S    -h-S-c-S
                           l
            """

            if MPS[isite].ndim == 3:
                path = [
                    ([0, 1], "fda, abc -> fdbc"),
                    ([2, 0], "fdbc, gdeb -> fcge"),
                    ([1, 0], "fcge, hec -> fgh"),
                ]
            elif MPS[isite].ndim == 4:
                path = [
                    ([0, 1], "fdla, abc -> fdlbc"),
                    ([2, 0], "fdlbc, gdeb -> flcge"),
                    ([1, 0], "flcge, helc -> fgh"),
                ]
            else:
                raise ValueError(
                    "MPS ndim at %d is not 3 or 4, got %s" % (isite, MPS[isite].ndim)
                )
            outtensor = multi_tensor_contract(
                path, MPSconj[isite], intensor, MPO[isite], MPS[isite]
            )

        return outtensor

    def write(self, domain, siteidx, tensor):
        self.virtual_disk[(domain, siteidx)] = tensor

    def read(self, domain: str, siteidx: int):
        return self.virtual_disk[(domain, siteidx)]


def select_basis(qnset, Sset, qnlist, Mmax, percent=0):
    """
    select basis according to Sset under qnlist requirement
    """

    # convert to dict
    basdic = dict()
    for i in range(len(qnset)):
        # clean quantum number outside qnlist
        if qnset[i] in qnlist:
            basdic[i] = [qnset[i], Sset[i]]

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

    return sidx


def updatemps(vset, sset, qnset, compset, nexciton, Mmax, percent=0):
    """
    select basis to construct new mps, and complementary mps
    vset, compset is the column vector
    """
    sidx = select_basis(qnset, sset, range(nexciton + 1), Mmax, percent=percent)
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
        compmps = Matrix(compmps)

    return Matrix(ms), mpsdim, mpsqn, compmps


def compressed_sum(mps_list, batchsize=5):
    assert len(mps_list) != 0
    mps_queue = deque(mps_list)
    while len(mps_queue) != 1:
        term_to_sum = []
        for i in range(min(batchsize, len(mps_queue))):
            term_to_sum.append(mps_queue.popleft())
        s = _sum(term_to_sum)
        mps_queue.append(s)
    return mps_queue[0]


def _sum(mps_list, compress=True):
    new_mps = reduce(lambda mps1, mps2: mps1.add(mps2), mps_list)
    if compress and not mps_list[0].compress_add:
        new_mps.canonicalise()
        new_mps.compress()
    return new_mps
