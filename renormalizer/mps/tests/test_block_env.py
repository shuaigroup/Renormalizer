# -*- coding: utf-8 -*-

import numpy as np
import pytest

from renormalizer.model import Model, h_qc
from renormalizer.mps import Mpo, Mps
from renormalizer.mps.gs import construct_mps_mpo, optimize_mps
from renormalizer.mps.hop_expr import hop_expr
from renormalizer.mps.lib import Environ
from renormalizer.mps.matrix import asxp, asnumpy, tensordot
from renormalizer.mps.block_env import (
    _arr,
    _mpo_lr_blocks,
    _mpo_lr_blocks_from_keys,
    _mpo_rl_blocks,
    _mpo_rl_blocks_from_keys,
    _qn_blocks,
    _symbolic_mpo_nonzero_keys,
    block_hdiag_two_site,
    block_hop_expr_two_site,
    contract_one_site_block,
)
from renormalizer.tests.parameter import holstein_model
from renormalizer.utils import CompressConfig
from renormalizer.utils.configs import OFS



@pytest.fixture
def numpy_block_kernels(request):
    """These private sparse kernels are intentionally NumPy-only.

    Optimization fallback tests below keep the caller-selected backend; only
    direct block-kernel tests enter this explicitly labelled host scope.
    """
    from renormalizer.backend.context import capture_backend, make_context
    request.node.user_properties.append(("execution_scope", "numpy-only block kernel"))
    with capture_backend(make_context("numpy").adapter):
        yield


def _max_env_diff(env1, env2, mps, mpo):
    diffs = []
    for idx in range(len(mps) - 1):
        a = env1.GetLR("L", idx, mps, mpo, method="Enviro")
        b = env2.GetLR("L", idx, mps, mpo, method="Enviro")
        diffs.append(np.max(np.abs(asnumpy(a) - asnumpy(b))))
    for idx in range(1, len(mps)):
        a = env1.GetLR("R", idx, mps, mpo, method="Enviro")
        b = env2.GetLR("R", idx, mps, mpo, method="Enviro")
        diffs.append(np.max(np.abs(asnumpy(a) - asnumpy(b))))
    return max(diffs)


def test_block_env_matches_dense_environ():
    mps, mpo = construct_mps_mpo(holstein_model, 10, 1)
    dense_env = Environ(mps, mpo)
    block_env = Environ(mps, mpo, use_block_env=True)
    assert _max_env_diff(dense_env, block_env, mps, mpo) < 1e-12


def test_block_env_optimization_matches_dense():
    for method in ["1site", "2site"]:
        mps0, mpo = construct_mps_mpo(holstein_model, 10, 1)
        energies = []
        for use_block_env in [False, True]:
            mps = mps0.copy()
            mps.optimize_config.procedure = [[10, 0.2], [10, 0]]
            mps.optimize_config.method = method
            mps.optimize_config.use_block_env = use_block_env
            energy, _ = optimize_mps(mps, mpo)
            energies.append(min(energy))
        assert abs(energies[0] - energies[1]) < 1e-10


def test_block_env_qc_hamiltonian_matches_dense_environ():
    mps, mpo = _toy_qc_mps_mpo()
    dense_env = Environ(mps, mpo)
    block_env = Environ(mps, mpo, use_block_env=True)
    assert _max_env_diff(dense_env, block_env, mps, mpo) < 1e-12


def _toy_qc_mps_mpo(bond_dim=12):
    h = np.array([[-1.0, 0.1], [0.1, -0.4]])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 0.7
    eri[1, 1, 1, 1] = 0.6
    eri[0, 0, 1, 1] = eri[1, 1, 0, 0] = 0.2
    eri[0, 1, 1, 0] = eri[1, 0, 0, 1] = 0.05
    h1e, h2e = h_qc.int_to_h(h, eri)
    basis, ham_terms = h_qc.qc_model(h1e, h2e)
    model = Model(basis, ham_terms)
    mpo = Mpo(model)
    mps = Mps.random(model, [1, 1], bond_dim, percent=1.0)
    return mps, mpo


def _chain_qc_mps_mpo(bond_dim=32):
    norbs = 6
    h = np.diag(np.linspace(-1.2, -0.2, norbs))
    h += 0.03 * (np.ones((norbs, norbs)) - np.eye(norbs))
    eri = np.zeros((norbs, norbs, norbs, norbs))
    for p in range(norbs):
        eri[p, p, p, p] = 0.6 + 0.02 * p
    for p in range(norbs):
        for q in range(norbs):
            if p == q:
                continue
            eri[p, p, q, q] = 0.12 / (1 + abs(p - q))
            eri[p, q, q, p] = 0.02 / (1 + abs(p - q))
    h1e, h2e = h_qc.int_to_h(h, eri)
    basis, ham_terms = h_qc.qc_model(h1e, h2e)
    model = Model(basis, ham_terms)
    mpo = Mpo(model)
    mps = Mps.random(model, [3, 3], bond_dim, percent=1.0)
    return mps, mpo


def _map_key_set(mpo_map):
    keys = set()
    for phys_key, entries in mpo_map.items():
        for qn, _idx, _block in entries:
            keys.add(tuple(phys_key) + (qn,))
    return keys


def test_symbolic_mpo_keys_reproduce_numeric_mpo_blocks():
    _mps, mpo = _chain_qc_mps_mpo()
    threshold = 1e-14
    for siteidx in range(len(mpo)):
        mo = _arr(mpo[siteidx])
        left_blocks = _qn_blocks(mpo.qn[siteidx])
        right_blocks = _qn_blocks(mpo.qn[siteidx + 1])
        symbolic_keys = _symbolic_mpo_nonzero_keys(mpo, siteidx, threshold)
        assert symbolic_keys is not None

        dense_lr = _mpo_lr_blocks(mo, left_blocks, right_blocks, threshold)
        symbolic_lr = _mpo_lr_blocks_from_keys(mo, left_blocks, right_blocks, symbolic_keys, threshold)
        assert _map_key_set(dense_lr) == _map_key_set(symbolic_lr)
        for key, entries in dense_lr.items():
            for (qn1, _idx1, block1), (qn2, _idx2, block2) in zip(entries, symbolic_lr[key]):
                assert qn1 == qn2
                assert np.max(np.abs(block1 - block2)) < 1e-14

        dense_rl = _mpo_rl_blocks(mo, left_blocks, right_blocks, threshold)
        symbolic_rl = _mpo_rl_blocks_from_keys(mo, left_blocks, right_blocks, symbolic_keys, threshold)
        assert _map_key_set(dense_rl) == _map_key_set(symbolic_rl)
        for key, entries in dense_rl.items():
            for (qn1, _idx1, block1), (qn2, _idx2, block2) in zip(entries, symbolic_rl[key]):
                assert qn1 == qn2
                assert np.max(np.abs(block1 - block2)) < 1e-14


def test_block_hop_two_site_matches_dense_hop(numpy_block_kernels):
    mps, mpo = _toy_qc_mps_mpo(bond_dim=16)
    mps.ensure_left_canonical()
    mps.move_qnidx(1)
    env = Environ(mps, mpo, use_block_env=True, block_env_min_bond_dim=0)
    cidx = [1, 2]
    left = env.read_raw("L", cidx[0] - 1)
    right = env.read_raw("R", cidx[1] + 1)
    center = asxp(tensordot(mps[cidx[0]], mps[cidx[1]], axes=1))
    dense_left = env.GetLR("L", cidx[0] - 1, mps, mpo, method="Enviro")
    dense_right = env.GetLR("R", cidx[1] + 1, mps, mpo, method="Enviro")
    dense_expr = hop_expr(dense_left, dense_right, [asxp(mpo[cidx[0]]), asxp(mpo[cidx[1]])], center.shape)
    qnbigl, qnbigr, qnmat = mps._get_big_qn(cidx)
    qn_mask = (qnmat == mps.qntot).all(axis=-1)
    block_expr = block_hop_expr_two_site(
        left,
        right,
        mpo[cidx[0]],
        mpo[cidx[1]],
        center.shape,
        qn_mask,
        mps.qn[cidx[0]],
        mps.qn[cidx[1]],
        mps.qn[cidx[1] + 1],
        mpo.qn[cidx[0]],
        mpo.qn[cidx[1]],
        mpo.qn[cidx[1] + 1],
    )
    diff = dense_expr(center)[qn_mask] - block_expr(center)[qn_mask]
    assert np.max(np.abs(diff)) < 1e-12
    packed_out = block_expr.apply_packed(np.asarray(center)[qn_mask])
    assert np.max(np.abs(packed_out - block_expr(center)[qn_mask])) < 1e-12


def test_center_block_layout_roundtrip(numpy_block_kernels):
    mps, mpo = _toy_qc_mps_mpo(bond_dim=16)
    mps.ensure_left_canonical()
    mps.move_qnidx(1)
    cidx = [1, 2]
    center = asnumpy(tensordot(mps[cidx[0]], mps[cidx[1]], axes=1))
    _qnbigl, _qnbigr, qnmat = mps._get_big_qn(cidx)
    qn_mask = (qnmat == mps.qntot).all(axis=-1)
    block_expr = block_hop_expr_two_site(
        Environ(mps, mpo, use_block_env=True, block_env_min_bond_dim=0).read_raw("L", cidx[0] - 1),
        Environ(mps, mpo, use_block_env=True, block_env_min_bond_dim=0).read_raw("R", cidx[1] + 1),
        mpo[cidx[0]],
        mpo[cidx[1]],
        center.shape,
        qn_mask,
        mps.qn[cidx[0]],
        mps.qn[cidx[1]],
        mps.qn[cidx[1] + 1],
        mpo.qn[cidx[0]],
        mpo.qn[cidx[1]],
        mpo.qn[cidx[1] + 1],
    )
    packed = block_expr.center_layout.pack(center)
    unpacked = block_expr.center_layout.unpack(packed)
    assert np.max(np.abs(unpacked[qn_mask] - center[qn_mask])) < 1e-14


def test_block_hdiag_two_site_matches_dense(numpy_block_kernels):
    mps, mpo = _toy_qc_mps_mpo(bond_dim=16)
    mps.ensure_left_canonical()
    mps.move_qnidx(1)
    env = Environ(mps, mpo, use_block_env=True, block_env_min_bond_dim=0)
    cidx = [1, 2]
    dense_left = env.GetLR("L", cidx[0] - 1, mps, mpo, method="Enviro")
    dense_right = env.GetLR("R", cidx[1] + 1, mps, mpo, method="Enviro")
    tmp_left = np.einsum("aba -> ba", dense_left)
    tmp_mo0 = np.einsum("abbc -> abc", np.asarray(mpo[cidx[0]]))
    tmp_mo1 = np.einsum("abbc -> abc", np.asarray(mpo[cidx[1]]))
    tmp_right = np.einsum("aba -> ba", dense_right)
    dense_hdiag = np.einsum("ba,bce,edg,gf->acdf", tmp_left, tmp_mo0, tmp_mo1, tmp_right)

    sparse_hdiag = block_hdiag_two_site(
        env.read_raw("L", cidx[0] - 1),
        env.read_raw("R", cidx[1] + 1),
        mpo[cidx[0]],
        mpo[cidx[1]],
        dense_hdiag.shape,
        mpo.qn[cidx[0]],
        mpo.qn[cidx[1]],
        mpo.qn[cidx[1] + 1],
    )
    assert np.max(np.abs(sparse_hdiag - dense_hdiag)) < 1e-12

def test_block_env_qc_optimization_converges_to_dense_energy():
    mps0, mpo = _chain_qc_mps_mpo()
    energies = []
    for use_block_env in [False, True]:
        mps = mps0.copy()
        mps.optimize_config.method = "2site"
        mps.optimize_config.algo = "davidson"
        mps.optimize_config.procedure = [[32, 0.5], [32, 0.2], [32, 0]]
        mps.optimize_config.use_block_env = use_block_env
        mps.optimize_config.block_env_min_bond_dim = 0
        energy, _ = optimize_mps(mps, mpo)
        energies.append(min(energy))
    assert abs(energies[0] - energies[1]) < 1e-8


def test_block_env_complex_mps_matches_dense_environ():
    mps, mpo = construct_mps_mpo(holstein_model, 10, 1)
    mps = mps.to_complex(inplace=True)
    mps[2] = mps[2] * (1.0 + 0.2j)
    dense_env = Environ(mps, mpo)
    block_env = Environ(mps, mpo, use_block_env=True)
    assert _max_env_diff(dense_env, block_env, mps, mpo) < 1e-12


def test_block_env_dense_fallback_matches_dense_environ():
    mps, mpo = construct_mps_mpo(holstein_model, 10, 1)
    dense_env = Environ(mps, mpo)
    block_env = Environ(mps, mpo, use_block_env=True, block_env_min_bond_dim=10**9)
    assert _max_env_diff(dense_env, block_env, mps, mpo) < 1e-12


def test_block_env_rejects_ofs():
    mps, mpo = construct_mps_mpo(holstein_model, 10, 1)
    mps.optimize_config.use_block_env = True
    mps.optimize_config.procedure = [[CompressConfig(ofs=OFS.ofs_s), 0]]
    with pytest.raises(NotImplementedError, match="use_block_env with OFS"):
        optimize_mps(mps, mpo)
