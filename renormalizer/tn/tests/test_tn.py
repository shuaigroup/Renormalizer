import numpy as np

from renormalizer import BasisHalfSpin, Model, Mpo, Mps, BasisMultiElectronVac, BasisMultiElectron
from renormalizer.model.model import heisenberg_ops
from renormalizer.tn.node import TreeNodeBasis
from renormalizer.tn.tree import TensorTreeOperator, TensorTreeState, TensorTreeEnviron, BasisTree
from renormalizer.tn.gs import optimize_tts
from renormalizer.tests.parameter import holstein_model


def test_tto():
    nspin = 7
    basis_list = [BasisHalfSpin(i) for i in range(nspin)]
    basis = BasisTree.binary(basis_list)
    assert basis.size == nspin
    ham_terms = heisenberg_ops(nspin)

    tto = TensorTreeOperator(basis, ham_terms)
    dense = tto.todense(basis_list)

    dense2 = Mpo(Model(basis_list, ham_terms)).todense()
    np.testing.assert_allclose(dense, dense2)


def test_tts():
    nspin = 7
    basis_list = [BasisHalfSpin(i) for i in range(nspin)]
    basis = BasisTree.binary(basis_list)
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


def test_gs_heisenberg():
    nspin = 8
    basis_list = [BasisHalfSpin(i) for i in range(nspin)]
    basis_tree = BasisTree.linear(basis_list)
    ham_terms = heisenberg_ops(4)
    condition = {1:1, 3:1}
    tts = TensorTreeState(basis_tree, condition)
    tto = TensorTreeOperator(basis_tree, ham_terms)
    m = 20
    e1 = optimize_tts(tts, tto, m)
    h = tto.todense()
    e2 = np.linalg.eigh(h)[0][0]
    np.testing.assert_allclose(e1, e2)


def test_gs_holstein():
    model = holstein_model.switch_scheme(4)
    node_list = [TreeNodeBasis(basis) for basis in model.basis]
    root = node_list.pop(2)
    assert isinstance(root.basis_set, BasisMultiElectronVac)
    # does not support QN so use another basis set
    root = TreeNodeBasis(BasisMultiElectron(root.basis_set.dofs, [0]*len(root.basis_set.dofs)))
    assert len(node_list) == 6
    for i in range(3):
        root.add_child(node_list[2*i])
        node_list[2*i].add_child(node_list[2*i+1])
    basis = BasisTree(root)
    tts = TensorTreeState(basis)
    tto = TensorTreeOperator(basis, model.ham_terms)
    m = 4
    e1 = optimize_tts(tts, tto, m)
    e2 = 0.08401412 + model.gs_zpe
    np.testing.assert_allclose(e1, e2)
