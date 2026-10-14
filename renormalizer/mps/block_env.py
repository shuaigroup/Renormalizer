# -*- coding: utf-8 -*-
"""Experimental QN-block sparse environment contractions.

This module keeps the implementation generic: it only uses the quantum-number
labels attached to MPS/MPO virtual bonds and does not assume a quantum-chemistry
operator structure.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

from renormalizer.backend.context import current_backend
from renormalizer.mps.matrix import asnumpy


Qn = Tuple[int, ...]
BlockKey = Tuple[Qn, Qn, Qn]
MpoLocalKey = Tuple[Qn, int, int, Qn]


@dataclass
class BlockEnvData:
    blocks: Dict[BlockKey, np.ndarray]
    bra_qn: np.ndarray
    mpo_qn: np.ndarray
    ket_qn: np.ndarray
    dtype: np.dtype

    def to_dense(self):
        bra_blocks = _qn_blocks(self.bra_qn)
        mpo_blocks = _qn_blocks(self.mpo_qn)
        ket_blocks = _qn_blocks(self.ket_qn)
        out = np.zeros(
            (len(self.bra_qn), len(self.mpo_qn), len(self.ket_qn)),
            dtype=self.dtype,
        )
        for (bra_qn, mpo_qn, ket_qn), block in self.blocks.items():
            out[np.ix_(bra_blocks[bra_qn], mpo_blocks[mpo_qn], ket_blocks[ket_qn])] += block
        return out


@dataclass(frozen=True)
class CenterBlock:
    lidx: np.ndarray
    ridx: np.ndarray
    vec_idx: np.ndarray
    shape: Tuple[int, int]


@dataclass(frozen=True)
class PackedHopConnection:
    in_vec_idx: np.ndarray
    in_shape: Tuple[int, int]
    left_half: np.ndarray
    rblock: np.ndarray


@dataclass(frozen=True)
class PackedOutputBlock:
    out_vec_idx: np.ndarray
    out_shape: Tuple[int, int]
    connections: Tuple[PackedHopConnection, ...]


class CenterBlockLayout:
    """Map active two-site center vector segments to QN/physical blocks."""

    def __init__(self, qn_mask: np.ndarray, left_qn: np.ndarray, right_qn: np.ndarray):
        self.qn_mask = np.asarray(qn_mask)
        self.shape = self.qn_mask.shape
        self.size = int(np.count_nonzero(self.qn_mask))
        flat_rank = np.full(self.qn_mask.size, -1, dtype=int)
        flat_rank[np.flatnonzero(self.qn_mask.ravel())] = np.arange(self.size)

        left_blocks = _qn_blocks(left_qn)
        right_blocks = _qn_blocks(right_qn)
        self.blocks: Dict[Tuple[Qn, int, int, Qn], CenterBlock] = {}
        for lqn, lidx in left_blocks.items():
            for rqn, ridx in right_blocks.items():
                for p0 in range(self.shape[1]):
                    for p1 in range(self.shape[2]):
                        mask = self.qn_mask[np.ix_(lidx, [p0], [p1], ridx)][:, 0, 0, :]
                        if not np.any(mask):
                            continue
                        if not np.all(mask):
                            raise NotImplementedError("partial QN blocks are not supported")
                        flat_idx = np.ravel_multi_index(
                            np.ix_(lidx, [p0], [p1], ridx),
                            self.shape,
                        ).reshape(-1)
                        vec_idx = flat_rank[flat_idx]
                        self.blocks[(lqn, p0, p1, rqn)] = CenterBlock(
                            lidx,
                            ridx,
                            vec_idx,
                            (len(lidx), len(ridx)),
                        )

    def get(self, lqn: Qn, p0: int, p1: int, rqn: Qn) -> Optional[CenterBlock]:
        return self.blocks.get((lqn, p0, p1, rqn))

    def unpack(self, vec: np.ndarray):
        out = np.zeros(self.shape, dtype=vec.dtype)
        for key, block in self.blocks.items():
            _lqn, p0, p1, _rqn = key
            out[np.ix_(block.lidx, [p0], [p1], block.ridx)] = vec[block.vec_idx].reshape(block.shape)[:, None, None, :]
        return out

    def pack(self, center: np.ndarray):
        out = np.empty(self.size, dtype=center.dtype)
        for key, block in self.blocks.items():
            _lqn, p0, p1, _rqn = key
            out[block.vec_idx] = center[np.ix_(block.lidx, [p0], [p1], block.ridx)][:, 0, 0, :].reshape(-1)
        return out


def _arr(x: Any) -> np.ndarray:
    return np.asarray(x.array if hasattr(x, "array") else asnumpy(x))


def _qn_blocks(qn: np.ndarray) -> Dict[Qn, np.ndarray]:
    groups = defaultdict(list)
    for i, item in enumerate(np.asarray(qn)):
        groups[tuple(int(x) for x in item)].append(i)
    return {key: np.asarray(val, dtype=int) for key, val in groups.items()}


def _nonzero(x: np.ndarray, threshold: float) -> bool:
    return bool(np.count_nonzero(np.abs(x) > threshold))


def _symbolic_mpo_nonzero_keys(mpo, siteidx: int, threshold: float) -> Optional[Set[MpoLocalKey]]:
    """Return QN/physical local MPO keys allowed by the symbolic MPO.

    The numeric MPO tensor groups virtual indices by QN.  This helper uses the
    already-built symbolic local MPO to avoid scanning impossible local operator
    entries.  It returns QN-level keys, so numeric cancellations inside the QN
    block are still checked by the block extraction helpers below.
    """
    symbolic_mpo = getattr(mpo, "symbolic_mpo", None)
    model = getattr(mpo, "model", None)
    if symbolic_mpo is None or model is None:
        return None
    try:
        symbolic_mo = symbolic_mpo[siteidx]
        basis = model.basis[siteidx]
        qn_left = np.asarray(mpo.qn[siteidx])
        qn_right = np.asarray(mpo.qn[siteidx + 1])
    except (AttributeError, IndexError, TypeError):
        return None

    keys: Set[MpoLocalKey] = set()
    for (left_idx, right_idx), terms in np.ndenumerate(symbolic_mo):
        if not terms:
            continue
        lqn = tuple(int(x) for x in qn_left[left_idx])
        rqn = tuple(int(x) for x in qn_right[right_idx])
        local = None
        for term in terms:
            mat = np.asarray(basis.op_mat(term))
            local = mat if local is None else local + mat
        if local is None:
            continue
        for pbra, pket in np.argwhere(np.abs(local) > threshold):
            keys.add((lqn, int(pbra), int(pket), rqn))
    return keys


class BlockEnvCache:
    """Cache static MPO block decompositions for block-sparse environments."""

    def __init__(self, mpo, threshold: float = 1e-14):
        self.mpo = mpo
        self.threshold = threshold
        self._mpo_lr = {}
        self._mpo_rl = {}
        self._symbolic_keys = {}

    def _key(self, siteidx: int):
        left = tuple(map(tuple, np.asarray(self.mpo.qn[siteidx]).tolist()))
        right = tuple(map(tuple, np.asarray(self.mpo.qn[siteidx + 1]).tolist()))
        return siteidx, left, right

    def mpo_lr_blocks(self, siteidx: int):
        key = self._key(siteidx)
        if key not in self._mpo_lr:
            mo = _arr(self.mpo[siteidx])
            left_blocks = _qn_blocks(self.mpo.qn[siteidx])
            right_blocks = _qn_blocks(self.mpo.qn[siteidx + 1])
            symbolic_keys = self.symbolic_nonzero_keys(siteidx)
            if symbolic_keys is None:
                self._mpo_lr[key] = _mpo_lr_blocks(mo, left_blocks, right_blocks, self.threshold)
            else:
                self._mpo_lr[key] = _mpo_lr_blocks_from_keys(
                    mo, left_blocks, right_blocks, symbolic_keys, self.threshold
                )
        return self._mpo_lr[key]

    def mpo_rl_blocks(self, siteidx: int):
        key = self._key(siteidx)
        if key not in self._mpo_rl:
            mo = _arr(self.mpo[siteidx])
            left_blocks = _qn_blocks(self.mpo.qn[siteidx])
            right_blocks = _qn_blocks(self.mpo.qn[siteidx + 1])
            symbolic_keys = self.symbolic_nonzero_keys(siteidx)
            if symbolic_keys is None:
                self._mpo_rl[key] = _mpo_rl_blocks(mo, left_blocks, right_blocks, self.threshold)
            else:
                self._mpo_rl[key] = _mpo_rl_blocks_from_keys(
                    mo, left_blocks, right_blocks, symbolic_keys, self.threshold
                )
        return self._mpo_rl[key]

    def symbolic_nonzero_keys(self, siteidx: int) -> Optional[Set[MpoLocalKey]]:
        key = self._key(siteidx)
        if key not in self._symbolic_keys:
            self._symbolic_keys[key] = _symbolic_mpo_nonzero_keys(self.mpo, siteidx, self.threshold)
        return self._symbolic_keys[key]


def _mps_lr_blocks(ms, left_blocks, right_blocks, threshold):
    by_left_phys = defaultdict(list)
    for lqn, lidx in left_blocks.items():
        for rqn, ridx in right_blocks.items():
            for phys in range(ms.shape[1]):
                block = ms[np.ix_(lidx, [phys], ridx)][:, 0, :]
                if _nonzero(block, threshold):
                    by_left_phys[(lqn, phys)].append((rqn, ridx, block))
    return by_left_phys


def _mps_rl_blocks(ms, left_blocks, right_blocks, threshold):
    by_right_phys = defaultdict(list)
    for lqn, lidx in left_blocks.items():
        for rqn, ridx in right_blocks.items():
            for phys in range(ms.shape[1]):
                block = ms[np.ix_(lidx, [phys], ridx)][:, 0, :]
                if _nonzero(block, threshold):
                    by_right_phys[(rqn, phys)].append((lqn, lidx, block))
    return by_right_phys


def _mpo_lr_blocks(mo, left_blocks, right_blocks, threshold):
    by_left_phys = defaultdict(list)
    for lqn, lidx in left_blocks.items():
        for rqn, ridx in right_blocks.items():
            for pbra in range(mo.shape[1]):
                for pket in range(mo.shape[2]):
                    block = mo[np.ix_(lidx, [pbra], [pket], ridx)][:, 0, 0, :]
                    if _nonzero(block, threshold):
                        by_left_phys[(lqn, pbra, pket)].append((rqn, ridx, block))
    return by_left_phys


def _mpo_lr_blocks_from_keys(mo, left_blocks, right_blocks, keys, threshold):
    by_left_phys = defaultdict(list)
    seen = set()
    for lqn, pbra, pket, rqn in keys:
        item_key = (lqn, int(pbra), int(pket), rqn)
        if item_key in seen:
            continue
        seen.add(item_key)
        lidx = left_blocks.get(lqn)
        ridx = right_blocks.get(rqn)
        if lidx is None or ridx is None:
            continue
        block = mo[np.ix_(lidx, [pbra], [pket], ridx)][:, 0, 0, :]
        if _nonzero(block, threshold):
            by_left_phys[(lqn, int(pbra), int(pket))].append((rqn, ridx, block))
    return by_left_phys


def _mpo_rl_blocks(mo, left_blocks, right_blocks, threshold):
    by_right_phys = defaultdict(list)
    for lqn, lidx in left_blocks.items():
        for rqn, ridx in right_blocks.items():
            for pbra in range(mo.shape[1]):
                for pket in range(mo.shape[2]):
                    block = mo[np.ix_(lidx, [pbra], [pket], ridx)][:, 0, 0, :]
                    if _nonzero(block, threshold):
                        by_right_phys[(rqn, pbra, pket)].append((lqn, lidx, block))
    return by_right_phys


def _mpo_rl_blocks_from_keys(mo, left_blocks, right_blocks, keys, threshold):
    by_right_phys = defaultdict(list)
    seen = set()
    for lqn, pbra, pket, rqn in keys:
        item_key = (lqn, int(pbra), int(pket), rqn)
        if item_key in seen:
            continue
        seen.add(item_key)
        lidx = left_blocks.get(lqn)
        ridx = right_blocks.get(rqn)
        if lidx is None or ridx is None:
            continue
        block = mo[np.ix_(lidx, [pbra], [pket], ridx)][:, 0, 0, :]
        if _nonzero(block, threshold):
            by_right_phys[(rqn, int(pbra), int(pket))].append((lqn, lidx, block))
    return by_right_phys


def _env_blocks(environ, bra_blocks, mpo_blocks, ket_blocks, threshold):
    blocks = []
    for aqn, aidx in bra_blocks.items():
        for bqn, bidx in mpo_blocks.items():
            for cqn, cidx in ket_blocks.items():
                block = environ[np.ix_(aidx, bidx, cidx)]
                if _nonzero(block, threshold):
                    blocks.append((aqn, bqn, cqn, block))
    return blocks


def dense_to_block_env(environ, bra_qn, mpo_qn, ket_qn, threshold=1e-14):
    environ = _arr(environ)
    blocks = {}
    for aqn, bqn, cqn, block in _env_blocks(
        environ,
        _qn_blocks(bra_qn),
        _qn_blocks(mpo_qn),
        _qn_blocks(ket_qn),
        threshold,
    ):
        blocks[(aqn, bqn, cqn)] = block.copy()
    return BlockEnvData(blocks, np.asarray(bra_qn), np.asarray(mpo_qn), np.asarray(ket_qn), environ.dtype)


def supports_block_env(ms, mo) -> bool:
    # These kernels use NumPy directly. Respect per-call backend capture rather
    # than the legacy import-time GPU flag, and reject before host conversion.
    return current_backend().name == "numpy" and _arr(ms).ndim == 3 and _arr(mo).ndim == 4


def _contract_l(eblock, ablock, oblock, kblock):
    """Contract ``abc,af,bg,ch->fgh`` without per-call path optimization."""
    a, b, c = eblock.shape
    f = ablock.shape[1]
    g = oblock.shape[1]
    h = kblock.shape[1]
    tmp = ablock.T @ eblock.reshape(a, b * c)
    tmp = tmp.reshape(f, b, c)
    tmp = tmp.transpose(0, 2, 1).reshape(f * c, b) @ oblock
    tmp = tmp.reshape(f, c, g).transpose(0, 2, 1)
    return (tmp.reshape(f * g, c) @ kblock).reshape(f, g, h)


def _contract_r(eblock, ablock, oblock, kblock):
    """Contract ``abc,fa,gb,hc->fgh`` without per-call path optimization."""
    a, b, c = eblock.shape
    f = ablock.shape[0]
    g = oblock.shape[0]
    h = kblock.shape[0]
    tmp = ablock @ eblock.reshape(a, b * c)
    tmp = tmp.reshape(f, b, c)
    tmp = tmp.transpose(0, 2, 1).reshape(f * c, b) @ oblock.T
    tmp = tmp.reshape(f, c, g).transpose(0, 2, 1)
    return (tmp.reshape(f * g, c) @ kblock.T).reshape(f, g, h)


def _center_block_indices(left_qn, right_qn, phys0: int, phys1: int, qn_mask: np.ndarray):
    left_blocks = _qn_blocks(left_qn)
    right_blocks = _qn_blocks(right_qn)
    by_left_phys = defaultdict(list)
    for lqn, lidx in left_blocks.items():
        for rqn, ridx in right_blocks.items():
            for p0 in range(phys0):
                for p1 in range(phys1):
                    mask = qn_mask[np.ix_(lidx, [p0], [p1], ridx)][:, 0, 0, :]
                    if np.any(mask):
                        by_left_phys[(lqn, p0, p1)].append((rqn, lidx, ridx))
    return by_left_phys


def _contract_hop_two_site(lblock, op_chain, rblock, cblock):
    """Contract ``abc,bf,fj,ljk,ck->al`` with explicit GEMM steps."""
    a, b, c = lblock.shape
    l, j, k = rblock.shape
    left = (lblock.transpose(0, 2, 1).reshape(a * c, b) @ op_chain).reshape(a, c, j)
    right = (rblock.reshape(l * j, k) @ cblock.T).reshape(l, j, c)
    return left.reshape(a, c * j) @ right.transpose(0, 2, 1).reshape(l, c * j).T


def _precompute_left_half(lblock, op_chain):
    a, b, c = lblock.shape
    j = op_chain.shape[1]
    return (lblock.transpose(0, 2, 1).reshape(a * c, b) @ op_chain).reshape(a, c, j)


def _thin_hop_operator(lblock, op_chain, rblock, threshold):
    row_mask = np.any(np.abs(op_chain) > threshold, axis=1)
    col_mask = np.any(np.abs(op_chain) > threshold, axis=0)
    if not np.any(row_mask) or not np.any(col_mask):
        return None
    if np.all(row_mask) and np.all(col_mask):
        return lblock, op_chain, rblock
    return (
        lblock[:, row_mask, :],
        op_chain[np.ix_(row_mask, col_mask)],
        rblock[:, col_mask, :],
    )


def _contract_hop_two_site_lhalf(left_half, rblock, cblock):
    """Contract two-site hop with precomputed ``abc,bj->acj`` left half."""
    a, c, j = left_half.shape
    l, _j, k = rblock.shape
    assert j == _j
    right = (rblock.reshape(l * j, k) @ cblock.T).reshape(l, j, c)
    return left_half.reshape(a, c * j) @ right.transpose(0, 2, 1).reshape(l, c * j).T


def _env_diag_blocks(env: BlockEnvData):
    by_mpo_qn = defaultdict(list)
    qn_blocks = _qn_blocks(env.bra_qn)
    for (bra_qn, mpo_qn, ket_qn), block in env.blocks.items():
        if bra_qn != ket_qn:
            continue
        idx = qn_blocks[bra_qn]
        diag = block[np.arange(len(idx)), :, np.arange(len(idx))]
        by_mpo_qn[mpo_qn].append((bra_qn, idx, diag))
    return by_mpo_qn


def _mpo_diag_chains(mo0, mo1, mpo_qn_left, mpo_qn_mid, mpo_qn_right, phys0: int, phys1: int, threshold: float):
    mo0 = _arr(mo0)
    mo1 = _arr(mo1)
    left_blocks = _qn_blocks(mpo_qn_left)
    mid_blocks = _qn_blocks(mpo_qn_mid)
    right_blocks = _qn_blocks(mpo_qn_right)
    chains = {}
    for bqn, bidx in left_blocks.items():
        for c in range(phys0):
            for eqn, eidx in mid_blocks.items():
                w0 = mo0[np.ix_(bidx, [c], [c], eidx)][:, 0, 0, :]
                if not _nonzero(w0, threshold):
                    continue
                for d in range(phys1):
                    for gqn, gidx in right_blocks.items():
                        w1 = mo1[np.ix_(eidx, [d], [d], gidx)][:, 0, 0, :]
                        if not _nonzero(w1, threshold):
                            continue
                        key = (bqn, c, d, gqn)
                        chain = w0 @ w1
                        if key in chains:
                            chains[key] += chain
                        else:
                            chains[key] = chain.copy()
    return chains


def block_hdiag_two_site(
    left_env: BlockEnvData,
    right_env: BlockEnvData,
    mo0,
    mo1,
    cshape,
    mpo_qn_left,
    mpo_qn_mid,
    mpo_qn_right,
    threshold: float = 1e-14,
):
    """Block-sparse two-site effective-Hamiltonian diagonal.

    This computes the same tensor as the dense path
    ``L[a,b,a] W0[b,c,c,e] W1[e,d,d,g] R[f,g,f]``.
    """
    if not isinstance(left_env, BlockEnvData) or not isinstance(right_env, BlockEnvData):
        raise TypeError("block_hdiag_two_site requires block-sparse left and right environments")
    cshape = tuple(cshape)
    dtype = np.result_type(left_env.dtype, right_env.dtype, _arr(mo0), _arr(mo1))
    hdiag = np.zeros(cshape, dtype=dtype)
    left_diag_by_mpo = _env_diag_blocks(left_env)
    right_diag_by_mpo = _env_diag_blocks(right_env)
    chains = _mpo_diag_chains(
        mo0,
        mo1,
        mpo_qn_left,
        mpo_qn_mid,
        mpo_qn_right,
        cshape[1],
        cshape[2],
        threshold,
    )
    for (bqn, c, d, gqn), chain in chains.items():
        left_items = left_diag_by_mpo.get(bqn, [])
        right_items = right_diag_by_mpo.get(gqn, [])
        if not left_items or not right_items:
            continue
        for _aqn, aidx, left_diag in left_items:
            left_chain = left_diag @ chain
            for _fqn, fidx, right_diag in right_items:
                hdiag[np.ix_(aidx, [c], [d], fidx)] += (left_chain @ right_diag.T)[:, None, None, :]
    return hdiag


class BlockHop2Site:
    """QN-block sparse two-site effective-Hamiltonian matvec.

    The callable returns the same dense tensor shape as ``hop_expr``. Vector
    masking and Davidson remain unchanged in ``gs.py``.
    """

    def __init__(
        self,
        left_env: BlockEnvData,
        right_env: BlockEnvData,
        mo0,
        mo1,
        cshape,
        qn_mask,
        mps_qn_left,
        mps_qn_mid,
        mps_qn_right,
        mpo_qn_left,
        mpo_qn_mid,
        mpo_qn_right,
        threshold: float = 1e-14,
        op0_keys=None,
        op1_keys=None,
    ):
        if not isinstance(left_env, BlockEnvData) or not isinstance(right_env, BlockEnvData):
            raise TypeError("BlockHop2Site requires block-sparse left and right environments")

        self.cshape = tuple(cshape)
        self.dtype = np.result_type(left_env.dtype, right_env.dtype, _arr(mo0), _arr(mo1))

        op0_left_blocks = _qn_blocks(mpo_qn_left)
        op0_right_blocks = _qn_blocks(mpo_qn_mid)
        if op0_keys is None:
            op0_map = _mpo_lr_blocks(_arr(mo0), op0_left_blocks, op0_right_blocks, threshold)
        else:
            op0_map = _mpo_lr_blocks_from_keys(_arr(mo0), op0_left_blocks, op0_right_blocks, op0_keys, threshold)
        op1_left_blocks = _qn_blocks(mpo_qn_mid)
        op1_right_blocks = _qn_blocks(mpo_qn_right)
        if op1_keys is None:
            op1_map = _mpo_lr_blocks(_arr(mo1), op1_left_blocks, op1_right_blocks, threshold)
        else:
            op1_map = _mpo_lr_blocks_from_keys(_arr(mo1), op1_left_blocks, op1_right_blocks, op1_keys, threshold)
        self.center_layout = CenterBlockLayout(qn_mask, mps_qn_left, mps_qn_right)
        center_map = _center_block_indices(
            mps_qn_left,
            mps_qn_right,
            self.cshape[1],
            self.cshape[2],
            qn_mask,
        )

        right_by_key = defaultdict(list)
        for (lqn, jqn, kqn), block in right_env.blocks.items():
            right_by_key[(jqn, kqn)].append((lqn, block))

        out_left_blocks = _qn_blocks(mps_qn_left)
        out_right_blocks = _qn_blocks(mps_qn_right)
        center_groups = {}
        packed_output_groups = {}
        for (aqn, bqn, cqn), lblock in left_env.blocks.items():
            aidx = out_left_blocks[aqn]
            for d in range(self.cshape[1]):
                for e in range(self.cshape[1]):
                    op0s = op0_map.get((bqn, d, e), [])
                    if not op0s:
                        continue
                    for h in range(self.cshape[2]):
                        centers = center_map.get((cqn, e, h), [])
                        if not centers:
                            continue
                        for fqn, _fidx, op0 in op0s:
                            for g in range(self.cshape[2]):
                                op1s = op1_map.get((fqn, g, h), [])
                                if not op1s:
                                    continue
                                for kqn, c_lidx, c_ridx in centers:
                                    center_block = self.center_layout.get(cqn, e, h, kqn)
                                    if center_block is None:
                                        continue
                                    for jqn, _jidx, op1 in op1s:
                                        for lqn, rblock in right_by_key.get((jqn, kqn), []):
                                            out_block = self.center_layout.get(aqn, d, g, lqn)
                                            if out_block is None:
                                                continue
                                            op_chain = op0 @ op1
                                            thinned = _thin_hop_operator(lblock, op_chain, rblock, threshold)
                                            if thinned is None:
                                                continue
                                            lblock_thin, op_chain_thin, rblock_thin = thinned
                                            left_half = _precompute_left_half(lblock_thin, op_chain_thin)
                                            group_key = (id(c_lidx), e, h, id(c_ridx))
                                            group = center_groups.get(group_key)
                                            if group is None:
                                                group = (e, h, c_lidx, c_ridx, [])
                                                center_groups[group_key] = group
                                            group[4].append(
                                                (
                                                    np.ix_(aidx, [d], [g], out_right_blocks[lqn]),
                                                    left_half,
                                                    rblock_thin,
                                                )
                                            )
                                            packed_key = id(out_block.vec_idx)
                                            packed_group = packed_output_groups.get(packed_key)
                                            if packed_group is None:
                                                packed_group = (out_block.vec_idx, out_block.shape, [])
                                                packed_output_groups[packed_key] = packed_group
                                            packed_group[2].append(
                                                PackedHopConnection(
                                                    center_block.vec_idx,
                                                    center_block.shape,
                                                    left_half,
                                                    rblock_thin,
                                                )
                                            )
        self.center_groups = list(center_groups.values())
        self.packed_output_groups = [
            PackedOutputBlock(out_vec_idx, out_shape, tuple(connections))
            for out_vec_idx, out_shape, connections in packed_output_groups.values()
        ]

    def __call__(self, center):
        center = _arr(center)
        out = np.zeros(self.cshape, dtype=np.result_type(self.dtype, center))
        for e, h, c_lidx, c_ridx, contractions in self.center_groups:
            cblock = center[np.ix_(c_lidx, [e], [h], c_ridx)][:, 0, 0, :]
            for out_idx, left_half, rblock in contractions:
                out[out_idx] = out[out_idx] + _contract_hop_two_site_lhalf(
                    left_half,
                    rblock,
                    cblock,
                )[:, None, None, :]
        return out

    def apply_packed(self, center_vec):
        center_vec = np.asarray(center_vec)
        out = np.zeros(self.center_layout.size, dtype=np.result_type(self.dtype, center_vec))
        for out_group in self.packed_output_groups:
            out_block = np.zeros(out_group.out_shape, dtype=out.dtype)
            for conn in out_group.connections:
                cblock = center_vec[conn.in_vec_idx].reshape(conn.in_shape)
                out_block += _contract_hop_two_site_lhalf(
                    conn.left_half,
                    conn.rblock,
                    cblock,
                ).reshape(out_group.out_shape)
            out[out_group.out_vec_idx] += out_block.reshape(-1)
        return out


def block_hop_expr_two_site(*args, **kwargs):
    return BlockHop2Site(*args, **kwargs)


def contract_one_site_block(
    environ,
    ms,
    mo,
    domain: str,
    mps_qn_left,
    mps_qn_right,
    mpo_qn_left,
    mpo_qn_right,
    cache: BlockEnvCache = None,
    siteidx: int = None,
    threshold: float = 1e-14,
    ms_conj=None,
):
    """QN-block sparse equivalent of ``contract_one_site`` for ordinary MPS."""
    raw_environ = environ
    environ = None if isinstance(raw_environ, BlockEnvData) else _arr(raw_environ)
    ms = _arr(ms)
    mo = _arr(mo)
    if ms.ndim != 3 or mo.ndim != 4:
        raise NotImplementedError("block environment supports ordinary MPS/MPO tensors only")
    if ms_conj is None:
        ms_conj = ms.conj()
    else:
        ms_conj = _arr(ms_conj)

    ml = _qn_blocks(mps_qn_left)
    mr = _qn_blocks(mps_qn_right)
    ol = _qn_blocks(mpo_qn_left)
    or_ = _qn_blocks(mpo_qn_right)

    if domain == "L":
        out_dtype = np.result_type(raw_environ.dtype if isinstance(raw_environ, BlockEnvData) else environ, ms, mo, ms_conj)
        if isinstance(raw_environ, BlockEnvData):
            envs = [(aqn, bqn, cqn, block) for (aqn, bqn, cqn), block in raw_environ.blocks.items()]
        else:
            envs = _env_blocks(environ, ml, ol, ml, threshold)
        ket_map = _mps_lr_blocks(ms, ml, mr, threshold)
        bra_map = _mps_lr_blocks(ms_conj, ml, mr, threshold)
        if cache is not None and siteidx is not None:
            op_map = cache.mpo_lr_blocks(siteidx)
        else:
            op_map = _mpo_lr_blocks(mo, ol, or_, threshold)
        out_blocks = {}

        for aqn, bqn, cqn, eblock in envs:
            for pbra in range(ms.shape[1]):
                bras = bra_map.get((aqn, pbra), [])
                if not bras:
                    continue
                for pket in range(ms.shape[1]):
                    ops = op_map.get((bqn, pbra, pket), [])
                    kets = ket_map.get((cqn, pket), [])
                    if not ops or not kets:
                        continue
                    for fqn, fidx, ablock in bras:
                        for gqn, gidx, oblock in ops:
                            for hqn, hidx, kblock in kets:
                                key = (fqn, gqn, hqn)
                                if key not in out_blocks:
                                    out_blocks[key] = np.zeros((len(fidx), len(gidx), len(hidx)), dtype=out_dtype)
                                out_blocks[key] += _contract_l(eblock, ablock, oblock, kblock)
        return BlockEnvData(out_blocks, np.asarray(mps_qn_right), np.asarray(mpo_qn_right), np.asarray(mps_qn_right), np.dtype(out_dtype))
    elif domain == "R":
        out_dtype = np.result_type(raw_environ.dtype if isinstance(raw_environ, BlockEnvData) else environ, ms, mo, ms_conj)
        if isinstance(raw_environ, BlockEnvData):
            envs = [(aqn, bqn, cqn, block) for (aqn, bqn, cqn), block in raw_environ.blocks.items()]
        else:
            envs = _env_blocks(environ, mr, or_, mr, threshold)
        ket_map = _mps_rl_blocks(ms, ml, mr, threshold)
        bra_map = _mps_rl_blocks(ms_conj, ml, mr, threshold)
        if cache is not None and siteidx is not None:
            op_map = cache.mpo_rl_blocks(siteidx)
        else:
            op_map = _mpo_rl_blocks(mo, ol, or_, threshold)
        out_blocks = {}

        for aqn, bqn, cqn, eblock in envs:
            for pbra in range(ms.shape[1]):
                bras = bra_map.get((aqn, pbra), [])
                if not bras:
                    continue
                for pket in range(ms.shape[1]):
                    ops = op_map.get((bqn, pbra, pket), [])
                    kets = ket_map.get((cqn, pket), [])
                    if not ops or not kets:
                        continue
                    for fqn, fidx, ablock in bras:
                        for gqn, gidx, oblock in ops:
                            for hqn, hidx, kblock in kets:
                                key = (fqn, gqn, hqn)
                                if key not in out_blocks:
                                    out_blocks[key] = np.zeros((len(fidx), len(gidx), len(hidx)), dtype=out_dtype)
                                out_blocks[key] += _contract_r(eblock, ablock, oblock, kblock)
        return BlockEnvData(out_blocks, np.asarray(mps_qn_left), np.asarray(mpo_qn_left), np.asarray(mps_qn_left), np.dtype(out_dtype))
    else:
        raise ValueError(f"unknown domain {domain!r}")
