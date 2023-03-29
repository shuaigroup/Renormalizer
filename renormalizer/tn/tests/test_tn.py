import pytest

from renormalizer import BasisHalfSpin, Model, Mpo, Mps, Op
from renormalizer.mps.backend import np
from renormalizer.model.model import heisenberg_ops
from renormalizer.tn.node import TreeNodeBasis
from renormalizer.tn.tree import TensorTreeOperator, TensorTreeState, TensorTreeEnviron
from renormalizer.tn.treebase import BasisTree
from renormalizer.tn.gs import optimize_tts
from renormalizer.tn.time_evolution import evolve
from renormalizer.tests.parameter import holstein_model
from renormalizer.tests.parameter_exact import model
from renormalizer.mps.tests.test_evolve import qutip_expectations, QUTIP_STEP


def multi_basis_tree(basis_list):
    #         3
    #        / \
    #       2
    #    /     \
    #  0,1    4, 5, 6
    node1 = TreeNodeBasis([basis_list[0], basis_list[1]])
    node2 = TreeNodeBasis([basis_list[2]])
    node3 = TreeNodeBasis([basis_list[3]])
    node4 = TreeNodeBasis([basis_list[4], basis_list[5], basis_list[6]])
    node3.add_child(node2)
    node2.add_child(node1)
    node2.add_child(node4)
    basis = BasisTree(node3)
    return basis


@pytest.mark.parametrize("multi_basis", [True, False])
def test_tto(multi_basis):
    nspin = 7
    basis_list = [BasisHalfSpin(i) for i in range(nspin)]
    if not multi_basis:
        basis = BasisTree.binary(basis_list)
        assert basis.size == nspin
    else:
        basis = multi_basis_tree(basis_list)

    ham_terms = heisenberg_ops(nspin)

    tto = TensorTreeOperator(basis, ham_terms)
    dense = tto.todense(basis_list)

    dense2 = Mpo(Model(basis_list, ham_terms)).todense()
    np.testing.assert_allclose(dense, dense2)


@pytest.mark.parametrize("multi_basis", [True, False])
def test_tts(multi_basis):
    nspin = 7
    basis_list = [BasisHalfSpin(i) for i in range(nspin)]
    if not multi_basis:
        basis = BasisTree.binary(basis_list)
        assert basis.size == nspin
    else:
        basis = multi_basis_tree(basis_list)
    ham_terms = heisenberg_ops(nspin)
    condition = {1:1, 3:1}
    tts = TensorTreeState(basis, condition)
    tto = TensorTreeOperator(basis, ham_terms)
    e1 = tts.expectation(tto)
    model = Model([BasisHalfSpin(i) for i in range(nspin)], ham_terms)
    mps = Mps.hartree_product_state(model, condition)
    mpo = Mpo(model)
    e2 = mps.expectation(mpo)
    np.testing.assert_allclose(e1, e2)
    env = TensorTreeEnviron(tts, tto)
    for node in env.node_list:
        for child, environ_child in zip(node.children, node.environ_children):
            e3 = environ_child.ravel() @ child.environ_parent.ravel()
            np.testing.assert_allclose(e3, e2)


@pytest.mark.parametrize("multi_basis", [True, False])
@pytest.mark.parametrize("ite", [False, True])
def test_gs_heisenberg(multi_basis, ite):
    nspin = 7
    basis_list = [BasisHalfSpin(i) for i in range(nspin)]
    if not multi_basis:
        basis_tree = BasisTree.binary(basis_list)
        assert basis_tree.size == nspin
    else:
        basis_tree = multi_basis_tree(basis_list)
    ham_terms = heisenberg_ops(4)
    tts = TensorTreeState.random(basis_tree, qntot=0, m_max=20)
    tto = TensorTreeOperator(basis_tree, ham_terms)
    if not ite:
        e1 = optimize_tts(tts, tto)
        e1 = min(e1)
    else:
        # imaginary time evolution for the ground state
        for i in range(10):
            tts.check_canonical()
            tts = evolve(tts, tto, -2j)
            tts.scale(1 / tts.tts_norm, inplace=True)
        e1 = tts.expectation(tto)
    h = tto.todense()
    e2 = np.linalg.eigh(h)[0][0]
    np.testing.assert_allclose(e1, e2)


@pytest.mark.parametrize("scheme", [3, 4])
def test_gs_holstein(scheme):
    if scheme == 3:
        model = holstein_model
        node_list = [TreeNodeBasis([basis]) for basis in model.basis]
        root = node_list[3]
        root.add_child(node_list[0])
        root.add_child(node_list[6])
        for i in range(3):
            node_list[3 * i].add_child(node_list[3 * i + 1])
            node_list[3 * i + 1].add_child(node_list[3 * i + 2])
    else:
        assert scheme == 4
        model = holstein_model.switch_scheme(4)
        node_list = [TreeNodeBasis([basis]) for basis in model.basis]
        root = node_list.pop(2)
        assert len(node_list) == 6
        for i in range(3):
            root.add_child(node_list[2*i])
            node_list[2*i].add_child(node_list[2*i+1])
    basis = BasisTree(root)
    m = 4
    tts = TensorTreeState.random(basis, qntot=1, m_max=m)
    tto = TensorTreeOperator(basis, model.ham_terms)
    procedure = [[m, 0.4], [m, 0.2], [m, 0.1], [m, 0], [m, 0]]
    e1 = optimize_tts(tts, tto, procedure)
    e2 = 0.08401412 + model.gs_zpe
    np.testing.assert_allclose(min(e1), e2)


@pytest.mark.parametrize("multi_basis", [True, False])
def test_add(multi_basis):
    nspin = 7
    basis_list = [BasisHalfSpin(i) for i in range(nspin)]
    if not multi_basis:
        basis_tree = BasisTree.binary(basis_list)
        assert basis_tree.size == nspin
    else:
        basis_tree = multi_basis_tree(basis_list)
    tts1 = TensorTreeState.random(basis_tree, qntot=0, m_max=4)
    tts2 = TensorTreeState.random(basis_tree, qntot=0, m_max=2)
    tts3 = tts1.add(tts2)
    s1 = tts1.todense()
    s2 = tts2.todense()
    s3 = tts3.todense()
    np.testing.assert_allclose(s1 + s2, s3)


@pytest.mark.parametrize("geometry", ["chain", "tree"])
def test_vmf(geometry):
    if geometry == "chain":
        basis = BasisTree.linear(model.basis)
    else:
        assert geometry == "tree"
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
    tto = TensorTreeOperator(basis, model.ham_terms)
    op_n_list = [TensorTreeOperator(basis, [Op(r"a^\dagger a", i)]) for i in range(3)]

    tts = TensorTreeState(basis, {0: 1})
    # expand bond dimension
    tts = tts + tts.random(basis, 1, 5).scale(1e-5, inplace=True)
    tts.canonicalise()

    tau = 0.5
    final_time = 2
    expectations = [[tts.expectation(o) for o in op_n_list]]
    for i in range(round(final_time / tau)):
        tts = evolve(tts, tto, tau)
        es = [tts.expectation(o) for o in op_n_list]
        expectations.append(es)
    qutip_end = round(final_time / QUTIP_STEP) + 1
    qutip_interval = round(tau / QUTIP_STEP)
    np.testing.assert_allclose(expectations, qutip_expectations[:qutip_end:qutip_interval], atol=5e-4)
