import os
from  typing import List

import pytest

from renormalizer import Op, Quantity
from renormalizer.mps.backend import np
from renormalizer.mps.mps import expand_bond_dimension_general
from renormalizer.mps.tests.test_evolve import QUTIP_STEP, qutip_expectations, init_mps
from renormalizer.tests.parameter_exact import model
from renormalizer.tests import parameter
from renormalizer.tn import BasisTree, TTNO, TTNS
from renormalizer.tn.tree import from_mps
from renormalizer.tn.node import TreeNodeBasis
from renormalizer.tn.utils_eph import max_entangled_ex
from renormalizer.utils import EvolveConfig, EvolveMethod, CompressConfig, CompressCriteria


def add_ttno_offset(ttns: TTNS, ttno: TTNO):
    e = ttns.expectation(ttno)
    ham_terms = ttno.terms.copy()
    ham_terms.append(ttns.basis.identity_op * (-e))
    return TTNO(ttno.basis, ham_terms)


def construct_ttns_and_ttno_chain():
    basis, ttns, ttno = from_mps(init_mps)
    op_n_list = [TTNO(basis, [Op(r"a^\dagger a", i)]) for i in range(3)]
    ttno = add_ttno_offset(ttns, ttno)
    return ttns, ttno, op_n_list


def construct_ttns_and_ttno_tree():
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
    ttno = TTNO(basis, model.ham_terms)
    op_n_list = [TTNO(basis, [Op(r"a^\dagger a", i)]) for i in range(3)]
    ttns = TTNS(basis, {0: 1})
    ttno = add_ttno_offset(ttns, ttno)
    return ttns, ttno, op_n_list


def construct_ttns_and_ttno_tree_mctdh():
    basis = BasisTree.binary_mctdh(model.basis)
    op_n_list = [TTNO(basis, [Op(r"a^\dagger a", i)]) for i in range(3)]
    ttns = TTNS(basis, {0: 1})
    ttno = TTNO(basis, model.ham_terms)
    ttno = add_ttno_offset(ttns, ttno)
    return ttns, ttno, op_n_list


init_chain = construct_ttns_and_ttno_chain()
init_tree = construct_ttns_and_ttno_tree()
init_tree_mctdh = construct_ttns_and_ttno_tree_mctdh()


def check_result(ttns: TTNS, ttno: TTNO, time_step: float, final_time: float, op_n_list: List, atol: float=1e-4):
    expectations = [[ttns.expectation(o) for o in op_n_list]]
    for i in range(round(final_time / time_step)):
        ttns = ttns.evolve(ttno, time_step)
        es = [ttns.expectation(o) for o in op_n_list]
        expectations.append(es)
    expectations = np.array(expectations)
    qutip_end = round(final_time / QUTIP_STEP) + 1
    qutip_interval = round(time_step / QUTIP_STEP)
    # more strict than mcd (the error criteria used for mps tests)
    np.testing.assert_allclose(expectations, qutip_expectations[:qutip_end:qutip_interval], atol=atol)
    diff = np.max(np.abs(expectations - qutip_expectations[:qutip_end:qutip_interval]), axis=0)
    print(diff)
    return ttns


@pytest.mark.parametrize("ttns_and_ttno", [init_chain, init_tree, init_tree_mctdh])
def test_tdvp_vmf(ttns_and_ttno):
    ttns, ttno, op_n_list = ttns_and_ttno
    # expand bond dimension
    ttns = ttns + ttns.random(ttns.basis, 1, 5).scale(1e-5, inplace=True)
    ttns.canonicalise()
    ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_vmf, ivp_rtol=1e-4, ivp_atol=1e-7, force_ovlp=False)
    check_result(ttns, ttno, 0.5, 2, op_n_list)


@pytest.mark.parametrize("ttns_and_ttno", [init_chain, init_tree, init_tree_mctdh])
def test_pc(ttns_and_ttno):
    ttns, ttno, op_n_list = ttns_and_ttno
    ttns = ttns.copy()
    ttns.evolve_config = EvolveConfig(EvolveMethod.prop_and_compress_tdrk4)
    ttns.compress_config = CompressConfig(CompressCriteria.fixed)
    check_result(ttns, ttno, 0.2, 5, op_n_list, 5e-4)


@pytest.mark.parametrize("ttns_and_ttno", [init_chain, init_tree, init_tree_mctdh])
@pytest.mark.parametrize("method", [EvolveMethod.tdvp_ps, EvolveMethod.tdvp_ps2])
def test_tdvp_ps(ttns_and_ttno, method):
    ttns, ttno, op_n_list = ttns_and_ttno
    if ttns_and_ttno is init_chain:
        ttns = ttns.copy()
    else:
        # expand bond dimension
        ttns = ttns + ttns.random(ttns.basis, 1, 5).scale(1e-5, inplace=True)
        ttns.canonicalise()
    ttns.evolve_config = EvolveConfig(method)
    ttns.compress_config = CompressConfig(CompressCriteria.fixed)
    if method is EvolveMethod.tdvp_ps:
        check_result(ttns, ttno, 0.4, 5, op_n_list)
    else:
        assert method is EvolveMethod.tdvp_ps2
        check_result(ttns, ttno, 2, 10, op_n_list, 5e-4)


def test_thermalprop():
    # imaginary time evolution on the P space. Q space untouched
    holstein_model = parameter.holstein_model

    basis_tree = BasisTree.binary_mctdh(holstein_model.basis, contract_primitive=True)
    basis_tree2 = basis_tree.add_auxiliary_space()

    ttns = max_entangled_ex(basis_tree2)
    ttns.compress_config.bond_dim_max_value = 12
    ttno = TTNO(basis_tree, holstein_model.ham_terms)
    ttns = expand_bond_dimension_general(ttns, hint_mpo=ttno)
    ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)


    beta = Quantity(298, "K").to_beta()
    evolve_time = beta / 2j
    nsteps = 50
    dbeta = evolve_time / nsteps
    for i in range(nsteps):
        ttns.evolve(ttno, dbeta)
        e = ttns.expectation(ttno)

    ne_ttno_list = [TTNO(basis_tree, Op(r"a^\dagger a", b.dof)) for b in holstein_model.basis if b.is_electron]
    occ = [ttns.expectation(ttno) for ttno in ne_ttno_list]

    etot_std = 0.0853388 + parameter.holstein_model.gs_zpe
    occ_std = [0.20896541050347484, 0.35240029674394463, 0.4386342927525734]
    # used small M and large dt
    # as long as the result is not absurdly wrong, it's fine
    rtol = 5e-3
    assert np.allclose(occ, occ_std, rtol=rtol)
    assert np.allclose(e, etot_std, rtol=rtol)


@pytest.mark.parametrize("ttns_and_ttno", [init_chain, init_tree, init_tree_mctdh])
def test_save_load(ttns_and_ttno):
    ttns, ttno, op_n_list = ttns_and_ttno
    ttns = ttns + ttns.random(ttns.basis, 1, 5).scale(1e-5, inplace=True)
    ttns.canonicalise()
    tau = 0.5
    ttns1 = ttns.copy()
    for i in range(2):
        ttns1 = ttns1.evolve(ttno, tau)
    exp1 = [ttns1.expectation(o) for o in op_n_list]
    ttns2 = ttns.evolve(ttno, tau)
    fname = f"{id(ttns2)}.npz"
    ttns2.dump(fname)
    ttns2 = TTNS.load(ttns.basis, fname)
    ttns2 = ttns2.evolve(ttno, tau)
    assert ttns2.coeff == ttns1.coeff
    exp2 = [ttns2.expectation(o) for o in op_n_list]
    np.testing.assert_allclose(exp2, exp1, atol=1e-7)
    os.remove(fname)
