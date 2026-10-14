import pytest
from renormalizer.mps.matrix import asnumpy

from renormalizer import BasisHalfSpin, Model, Mpo, Mps, Op
from renormalizer import optimize_mps
from renormalizer.mps.backend import np
from renormalizer.model.basis import BasisDummy, BasisSHO
from renormalizer.model.model import heisenberg_ops
from renormalizer.tn.node import TreeNodeBasis
from renormalizer.tn.tree import TTNO, TTNS, TTNEnviron, from_mps
from renormalizer.tn.treebase import BasisTree
from renormalizer.tn.gs import optimize_ttns
from renormalizer.tests.parameter import holstein_model
from renormalizer.tests.parameter_exact import model
from renormalizer.utils import CompressConfig, CompressCriteria


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


nspin = 7
basis_list = [BasisHalfSpin(i) for i in range(nspin)]
basis_binary = BasisTree.binary(basis_list)
basis_multi_basis = multi_basis_tree(basis_list)


def test_zero_qn_basis_expands_to_multidimensional_qn_in_node():
    electron = BasisHalfSpin("e", sigmaqn=np.array([[0, 0], [1, 0]]))
    phonon = BasisSHO("v", 1.0, 4)
    node = TreeNodeBasis([electron, phonon])
    assert node.qn_size == 2
    np.testing.assert_array_equal(phonon.sigmaqn, np.zeros((4, 2), dtype=int))


def test_zero_qn_nodes_expand_to_multidimensional_qn_in_tree():
    electron = TreeNodeBasis(BasisHalfSpin("e", sigmaqn=np.array([[0, 0], [1, 0]])))
    phonon = TreeNodeBasis(BasisSHO("v", 1.0, 4))
    root = TreeNodeBasis(BasisDummy("root"))
    root.add_child([electron, phonon])
    basis = BasisTree(root)
    assert basis.qn_size == 2
    assert root.qn_size == 2
    assert phonon.qn_size == 2
    np.testing.assert_array_equal(root.basis_sets[0].sigmaqn, np.zeros((1, 2), dtype=int))
    np.testing.assert_array_equal(phonon.basis_sets[0].sigmaqn, np.zeros((4, 2), dtype=int))


def test_nonzero_qn_dimension_mismatch_still_raises():
    electron_1d = BasisHalfSpin("e1", sigmaqn=np.array([[0], [1]]))
    electron_2d = BasisHalfSpin("e2", sigmaqn=np.array([[0, 0], [1, 0]]))
    with pytest.raises(ValueError, match="Inconsistent quantum number size"):
        TreeNodeBasis([electron_1d, electron_2d])


def holstein_scheme3() -> BasisTree:
    model = holstein_model
    node_list = [TreeNodeBasis([basis]) for basis in model.basis]
    root = node_list[3]
    root.add_child(node_list[0])
    root.add_child(node_list[6])
    for i in range(3):
        node_list[3 * i].add_child(node_list[3 * i + 1])
        node_list[3 * i + 1].add_child(node_list[3 * i + 2])
    return BasisTree(root)


@pytest.mark.parametrize("basis", [basis_binary, basis_multi_basis])
def test_ttno(basis):
    ham_terms = heisenberg_ops(nspin)

    ttno = TTNO(basis, ham_terms)
    dense = ttno.todense(basis_list)

    dense2 = Mpo(Model(basis_list, ham_terms)).todense()
    np.testing.assert_allclose(dense, dense2, atol=1e-15)


@pytest.mark.parametrize("basis", [basis_binary, basis_multi_basis])
def test_ttns(basis):
    ham_terms = heisenberg_ops(nspin)
    condition = {1:1, 3:1}
    ttns = TTNS(basis, condition)
    ttno = TTNO(basis, ham_terms)
    e1 = ttns.expectation(ttno)
    model = Model([BasisHalfSpin(i) for i in range(nspin)], ham_terms)
    mps = Mps.hartree_product_state(model, condition)
    mpo = Mpo(model)
    e2 = mps.expectation(mpo)
    np.testing.assert_allclose(e1, e2)
    env = TTNEnviron(ttns, ttno)
    for node in env.node_list:
        for child, environ_child in zip(node.children, node.environ_children):
            e3 = environ_child.ravel() @ child.environ_parent.ravel()
            np.testing.assert_allclose(e3, e2)


@pytest.mark.parametrize("basis", [basis_binary, basis_multi_basis])
def test_push_cano(basis):
    ttns = TTNS.random(basis, 0, 5, 1)
    s1 = ttns.todense()
    ttns.push_cano_to_child(ttns.root, 0)
    s2 = ttns.todense()
    np.testing.assert_allclose(s2, s1)
    ttns.push_cano_to_parent(ttns.root.children[0])
    s3 = ttns.todense()
    np.testing.assert_allclose(s3, s1)


def test_from_mps():
    mps = Mps.random(model, 1, 10)
    mpo = Mpo(model)
    e_ref = mps.expectation(mpo)
    basis, ttns, ttno = from_mps(mps)
    e = ttns.expectation(ttno)
    np.testing.assert_allclose(e, e_ref)


@pytest.mark.parametrize("basis_tree", [basis_binary, basis_multi_basis])
@pytest.mark.parametrize("ite", [False, True])
def test_gs_heisenberg(basis_tree, ite):
    ham_terms = heisenberg_ops(4)
    ttns = TTNS.random(basis_tree, qntot=0, m_max=20)
    ttno = TTNO(basis_tree, ham_terms)
    if not ite:
        e1 = optimize_ttns(ttns, ttno)
        e1 = min(e1)
    else:
        # imaginary time evolution for the ground state
        for i in range(10):
            ttns.check_canonical()
            ttns = ttns.evolve(ttno, -2j)
        e1 = ttns.expectation(ttno)
    h = ttno.todense()
    e2 = np.linalg.eigh(h)[0][0]
    np.testing.assert_allclose(e1, e2)


@pytest.mark.parametrize("scheme", [3, 4])
@pytest.mark.parametrize("m_type", [int, list])
def test_gs_holstein(scheme, m_type):
    if scheme == 3:
        model = holstein_model
        basis = holstein_scheme3()
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
    ttns = TTNS.random(basis, qntot=1, m_max=m)
    ttno = TTNO(basis, model.ham_terms)
    if m_type == list:
        m = ttns.bond_dims
    procedure = [[m, 0.4], [m, 0.2], [m, 0.1], [m, 0], [m, 0]]
    e1 = optimize_ttns(ttns, ttno, procedure)
    e2 = 0.08401412 + model.gs_zpe
    np.testing.assert_allclose(min(e1), e2)


@pytest.mark.parametrize("basis_tree", [basis_binary, basis_multi_basis])
def test_add(basis_tree):
    ttns1 = TTNS.random(basis_tree, qntot=0, m_max=4)
    ttns2 = TTNS.random(basis_tree, qntot=0, m_max=2).scale(1j)
    ttns3 = ttns1.add(ttns2)
    s1 = ttns1.todense()
    s2 = ttns2.todense()
    assert np.iscomplexobj(s2)
    s3 = ttns3.todense()
    np.testing.assert_allclose(s1 + s2, s3)


@pytest.mark.parametrize("basis_tree", [basis_binary, basis_multi_basis])
def test_overlap(basis_tree):
    ttns1 = TTNS.random(basis_tree, qntot=0, m_max=4).scale(1 + 0.2j)
    ttns2 = TTNS.random(basis_tree, qntot=0, m_max=2).scale(0.7 - 0.3j)
    overlap = ttns1.overlap(ttns2)
    dense_overlap = ttns2.todense().ravel().conj() @ ttns1.todense().ravel()
    np.testing.assert_allclose(overlap, dense_overlap)


@pytest.mark.parametrize("basis_tree", [basis_binary, basis_multi_basis])
def test_transition_expectation(basis_tree):
    ttns1 = TTNS.random(basis_tree, qntot=0, m_max=4).scale(1 + 0.2j)
    ttns2 = TTNS.random(basis_tree, qntot=0, m_max=2).scale(0.7 - 0.3j)
    ttno = TTNO(basis_tree, heisenberg_ops(nspin))
    transition = ttns1.expectation(ttno, bra=ttns2)
    dense_transition = (
        ttns2.todense().ravel().conj()
        @ ttno.todense().reshape(2 ** nspin, 2 ** nspin)
        @ ttns1.todense().ravel()
    )
    np.testing.assert_allclose(transition, dense_transition)


@pytest.mark.parametrize("basis_tree", [basis_binary, basis_multi_basis])
def test_apply(basis_tree):
    ttns1 = TTNS.random(basis_tree, qntot=0, m_max=4)
    ttno = TTNO(basis_tree, heisenberg_ops(nspin))
    ttns2 = ttno.apply(ttns1)
    s1 = ttns1.todense()
    s2 = ttns2.todense()
    op = ttno.todense()
    np.testing.assert_allclose(s2.ravel(), op @ s1.ravel())


def test_compress():
    m1 = 5
    m2 = 4
    basis = holstein_scheme3()
    ttno = TTNO(basis, holstein_model.ham_terms)
    ttns = TTNS.random(basis, 1, m1)
    procedure1, procedure2 = [[[m, 0.4], [m, 0.2], [m, 0.1], [m, 0], [m, 0]] for m in [m1, m2]]
    optimize_ttns(ttns, ttno, procedure1)
    ttns2 = ttns.copy().compress(m2)
    optimize_ttns(ttns, ttno, procedure2)
    s1 = ttns.todense().ravel()
    s2 = ttns2.todense().ravel()

    np.testing.assert_allclose(np.abs(s1 @ s2), 1, atol=1e-5)

    ttns3 = ttns.copy()
    ttns3.compress_config = CompressConfig(CompressCriteria.fixed)
    ttns3.compress_config.max_dims = [1, 4, 4, 2, 4, 2, 4, 4, 2]
    ttns3.compress()
    assert ttns3.bond_dims == ttns3.compress_config.max_dims
    s3 = ttns3.todense().ravel()
    np.testing.assert_allclose(np.abs(s3 @ s1), 1, atol=1e-5)


@pytest.mark.parametrize("basis_tree", [basis_binary, basis_multi_basis])
def test_partial_ttno(basis_tree):
    # apply ttno on ttns when ttns has more dofs than ttno
    basis_tree2 = basis_tree.add_auxiliary_space()
    ttns = TTNS.random(basis_tree2, qntot=0, m_max=4)
    ttno = TTNO(basis_tree, heisenberg_ops(nspin))
    ttno2 = TTNO(basis_tree2, heisenberg_ops(nspin))
    # test apply
    new_ttns = ttno.apply(ttns)
    new_ttns2 = ttno2.apply(ttns)
    np.testing.assert_allclose(new_ttns.todense(), new_ttns2.todense())
    # test expectation
    e = ttns.expectation(ttno)
    e2 = ttns.expectation(ttno2)
    np.testing.assert_allclose(e, e2)

@pytest.mark.parametrize("basis_tree", [basis_binary, basis_multi_basis])
def test_1site_entropy(basis_tree):
    ttns = TTNS.random(basis_tree, 0, 5, 1)
    bond_entropy = ttns.calc_bond_entropy()
    site1_entropy = ttns.calc_1site_entropy()
    for i, node in enumerate(ttns):
        if node.is_leaf:
            np.testing.assert_allclose(bond_entropy[i], site1_entropy[i], atol=1e-10)


def test_rdm_entropy_holstein():
    # the Heisenberg model seem to do not have well-defined single-body expectations and two-body RDMs
    # so use Holstein model instead
    model = holstein_model
    basis = holstein_scheme3()
    m = 16
    ttns = TTNS.random(basis, qntot=1, m_max=m)
    ttno = TTNO(basis, model.ham_terms)
    mps = Mps.random(model, qntot=1, m_max=m)
    mpo = Mpo(model)
    procedure = [[m, 0.4], [m, 0.2], [m, 0.1], [m, 0], [m, 0]]
    e1 = optimize_ttns(ttns, ttno, procedure)
    e2 = 0.08401412 + model.gs_zpe
    np.testing.assert_allclose(min(e1), e2)
    optimize_mps(mps, mpo)

    mps_rdm_dict = mps.calc_1site_rdm()
    ttns_rdm_dict = ttns.calc_1dof_rdm()
    for i in range(len(mps)):
        dof = model.basis[i].dof
        np.testing.assert_allclose(asnumpy(mps_rdm_dict[i]), asnumpy(ttns_rdm_dict[dof]), atol=1e-3)

    mps_mutual_info = mps.calc_2site_mutual_entropy()
    mps_idx1, mps_idx2 = 1, 3
    dof1 = model.basis[mps_idx1].dof
    dof2 = model.basis[mps_idx2].dof
    ttns_mutual_infos,_ = ttns.calc_2dof_mutual_info((dof1, dof2))
    ttns_mutual_info = ttns_mutual_infos[(dof1, dof2)]
    np.testing.assert_allclose(ttns_mutual_info, mps_mutual_info[mps_idx1, mps_idx2], atol=1e-4)


@pytest.mark.parametrize("basis_tree", [basis_binary, basis_multi_basis])
@pytest.mark.parametrize("dofs", [(1, 5)])  # see `multi_basis_tree`
def test_2dof_rdm(basis_tree, dofs):
    m = 32
    ham_terms = heisenberg_ops(nspin)

    procedure = [[m, 0.4], [m, 0.2], [m, 0.1], [m, 0], [m, 0]]

    ttns = TTNS.random(basis_tree, 0, m, 1)
    ttno = TTNO(basis_tree, ham_terms)
    e1 = optimize_ttns(ttns, ttno, procedure)

    model = Model(basis_list, ham_terms)
    mps = Mps.random(model, 0, m, 1)
    mpo = Mpo(model)
    e2 = optimize_mps(mps, mpo)[0]
    np.testing.assert_allclose(min(e1), min(e2))

    # test 2 dof rdm
    dof1, dof2 = dofs

    rdm1 = ttns.calc_2dof_rdm((dof1, dof2))[(dof1, dof2)].reshape(4, 4)
    rdm2 = mps.calc_2site_rdm()[(dof1, dof2)].reshape(4, 4)
    #np.testing.assert_allclose(rdm1, rdm2, atol=1e-10)

    # Z0Z1
    op1 = np.diag([1, -1, -1, 1])
    np.testing.assert_allclose(np.trace(asnumpy(rdm1) @ op1), np.trace(asnumpy(rdm2) @ op1), atol=1e-7)

    # +0-1 + +1-0
    op2 = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    np.testing.assert_allclose(np.trace(asnumpy(rdm1) @ op2), np.trace(asnumpy(rdm2) @ op2), atol=1e-7)


@pytest.mark.parametrize("basis", [basis_binary, basis_multi_basis])
def test_print(basis):
    basis.print()
    ttns = TTNS.random(basis, 0, 5, 1)
    ttns.print_shape()
    ttns.print_vn_entropy()
