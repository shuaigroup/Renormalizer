from renormalizer import Op
from renormalizer.tn import BasisTree, TTNS, TTNO, TreeNodeTensor, copy_connection
from renormalizer.mps.backend import np


def max_entangled_ex(basis_tree: BasisTree):
    # basis_tree should have both P and Q space
    node_tensor_list = []
    for basis_node in basis_tree:
        nbas = basis_node.basis_sets[0].nbas
        if basis_node.n_sets == 1:
            # dummy
            assert nbas == 1
            tensor = np.ones([1] * len(basis_node.children) + [1, 1])
        elif basis_node.n_sets == 2 and basis_node.basis_sets[0].is_phonon:
            shape = [1] * len(basis_node.children) + [nbas, nbas, 1]
            tensor = np.eye(nbas).reshape(shape)
            tensor /= np.sqrt(nbas)
        elif basis_node.n_sets == 2 and basis_node.basis_sets[0].is_electron:
            shape = [1] * len(basis_node.children) + [nbas, nbas, 1]
            tensor = np.zeros(shape)
            tensor[..., 0, 0, 0] = 1
        else:
            assert False
        node = TreeNodeTensor(tensor, np.array([0]).reshape(1, basis_tree.qn_size))
        node_tensor_list.append(node)

    copy_connection(basis_tree.node_list, node_tensor_list)
    ttns = TTNS(basis_tree, root=node_tensor_list[0])
    ex_ops = []
    for b in basis_tree.basis_list:
        # skip Q space
        if isinstance(b.dof, tuple) and len(b.dof) == 2 and b.dof[0] == "Q":
            continue
        if b.is_electron:
            op = Op(r"a^\dagger a^\dagger", [b.dof, ("Q", b.dofs)], qn=[1, 0])
            ex_ops.append(op)
    ex_ttno = TTNO(basis_tree, ex_ops)
    ttns = ex_ttno.apply(ttns)
    ttns.normalize("ttns_and_coeff")
    return ttns
