import logging

import pytest
from  typing import List

from renormalizer import Op
from renormalizer.mps.backend import np
from renormalizer.mps.tests.test_evolve import QUTIP_STEP, qutip_expectations, init_mps
from renormalizer.tests.parameter_exact import model
from renormalizer.tn import BasisTree, TTNO, TTNS
from renormalizer.tn.tree import from_mps
from renormalizer.tn.node import TreeNodeBasis
from renormalizer.utils import EvolveConfig, EvolveMethod, CompressConfig, CompressCriteria


def add_tto_offset(tts: TTNS, tto: TTNO):
    e = tts.expectation(tto)
    ham_terms = tto.ham_terms.copy()
    ham_terms.append(tts.basis.identity_op * (-e))
    return TTNO(tto.basis, ham_terms)



def construct_tts_and_tto_chain():
    basis, tts, tto = from_mps(init_mps)
    op_n_list = [TTNO(basis, [Op(r"a^\dagger a", i)]) for i in range(3)]
    tto = add_tto_offset(tts, tto)
    return tts, tto, op_n_list


def construct_tts_and_tto_tree():
    node_list = [TreeNodeBasis([basis]) for basis in model.basis]
    # 0 - 2 - 4
    # |   |   |
    # 1   3   5
    root = node_list[2]
    root.add_child(node_list[0])
    root.add_child(node_list[3])
    root.add_child(node_list[4])
    node_list[0].add_child(node_list[1])
    node_list[4].add_child(node_list[5])
    basis = BasisTree(root)
    tto = TTNO(basis, model.ham_terms)
    op_n_list = [TTNO(basis, [Op(r"a^\dagger a", i)]) for i in range(3)]
    tts = TTNS(basis, {0: 1})
    tto = add_tto_offset(tts, tto)
    return tts, tto, op_n_list


init_chain = construct_tts_and_tto_chain()
init_tree = construct_tts_and_tto_tree()


def check_result(tts: TTNS, tto: TTNO, time_step: float, final_time: float, op_n_list: List, atol: float=1e-4):
    expectations = [[tts.expectation(o) for o in op_n_list]]
    for i in range(round(final_time / time_step)):
        tts = tts.evolve(tto, time_step)
        es = [tts.expectation(o) for o in op_n_list]
        expectations.append(es)
    expectations = np.array(expectations)
    qutip_end = round(final_time / QUTIP_STEP) + 1
    qutip_interval = round(time_step / QUTIP_STEP)
    # more strict than mcd (the error criteria used for mps tests)
    np.testing.assert_allclose(expectations, qutip_expectations[:qutip_end:qutip_interval], atol=atol)
    diff = np.max(np.abs(expectations - qutip_expectations[:qutip_end:qutip_interval]), axis=0)
    print(diff)
    return tts


@pytest.mark.parametrize("ttns_and_ttno", [init_chain, init_tree])
def test_tdvp_vmf(ttns_and_ttno):
    ttns, ttno, op_n_list = ttns_and_ttno
    # expand bond dimension
    ttns = ttns + ttns.random(ttns.basis, 1, 5).scale(1e-5, inplace=True)
    ttns.canonicalise()
    ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_vmf, ivp_rtol=1e-4, ivp_atol=1e-7, force_ovlp=False)
    check_result(ttns, ttno, 0.5, 2, op_n_list)


@pytest.mark.parametrize("tts_and_tto", [init_chain, init_tree])
def test_pc(tts_and_tto):
    tts, tto, op_n_list = tts_and_tto
    tts = tts.copy()
    tts.evolve_config = EvolveConfig(EvolveMethod.prop_and_compress_tdrk4)
    tts.compress_config = CompressConfig(CompressCriteria.fixed)
    check_result(tts, tto, 0.2, 5, op_n_list, 5e-4)


@pytest.mark.parametrize("ttns_and_ttno", [init_chain, init_tree])
def test_tdvp_ps(ttns_and_ttno):
    ttns, ttno, op_n_list = ttns_and_ttno
    if ttns_and_ttno is init_chain:
        ttns = ttns.copy()
    else:
        # expand bond dimension
        ttns = ttns + ttns.random(ttns.basis, 1, 5).scale(1e-5, inplace=True)
        ttns.canonicalise()
    ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    ttns.compress_config = CompressConfig(CompressCriteria.fixed)
    check_result(ttns, ttno, 0.4, 5, op_n_list)


@pytest.mark.parametrize("ttns_and_ttno", [init_chain, init_tree])
def test_tdvp_ps2(ttns_and_ttno):
    ttns, ttno, op_n_list = ttns_and_ttno
    ttns = ttns.copy()
    ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps2)
    ttns.compress_config = CompressConfig(CompressCriteria.fixed)
    check_result(ttns, ttno, 2, 10, op_n_list, 5e-4)
