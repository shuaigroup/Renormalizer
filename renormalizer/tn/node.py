from typing import List, Dict, Union, Sequence

from renormalizer.mps.backend import np, backend
from renormalizer.mps.matrix import asnumpy
from renormalizer.model.basis import BasisSet, BasisDummy


class TreeNode:
    def __init__(self):
        self.children: List[__class__] = []
        self.parent: __class__ = None

    def add_child(self, node: Union["TreeNode", Sequence["TreeNode"]]) -> "TreeNode":
        if isinstance(node, TreeNode):
            nodes = [node]
        else:
            nodes = node

        for node in nodes:
            if node.parent is not None:
                raise ValueError("Node already has parent")
            self.children.append(node)
            node.parent = self

        return self

    # alias
    add_children = add_child

    @property
    def ancestors(self) -> List:
        """
        Returns the list of ancestors of this node, including itself
        """
        ancestors = [self]
        current = self
        while current.parent is not None:
            ancestors.append(current.parent)
            current = current.parent
        return ancestors

    @property
    def idx_as_child(self) -> int:
        """
        Returns the index of this node as a child of its parent
        """
        assert self.parent
        return self.parent.children.index(self)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


# indices for dummy basis sets
DUMMY_IDX = 0


class TreeNodeBasis(TreeNode):
    # tree node whose data is basis sets
    def __init__(self, basis_sets: Union[BasisSet, List[BasisSet]]=None):
        """
        Tree node whose data is basis sets.

        Parameters
        ----------
        basis_sets: ``BasisSet`` instances or list of ``BasisSet`` instances.
            The basis sets and degrees of freedom to be included in this node.
            If a list of provided, then this node has multiple physical degrees of freedom,
            and the corresponding node in TTNS will have multiple physical indices.
            If no basis sets are provided, a dummy basis set is created.
            Hilbert space dimension for this dummy basis set is 1, that is,
            the node is not associated with any physical degrees of freedom.

        Examples
        --------
        >>> from renormalizer import BasisHalfSpin, BasisMultiElectronVac
        >>> from renormalizer.tn import TreeNodeBasis
        >>> node1 = TreeNodeBasis(BasisHalfSpin("spin"))
        >>> node1
        TreeNodeBasis(BasisHalfSpin(dof: spin, nbas: 2))
        >>> node1.dofs
        [('spin',)]
        >>> node2 = TreeNodeBasis([BasisHalfSpin("spin"), BasisMultiElectronVac(["e1", "e2"])])
        >>> node2.dofs
        [('spin',), ('e1', 'e2')]
        >>> node3 = TreeNodeBasis()
        >>> node3.basis_set
        BasisDummy(dof: ('Virtual DOF', 0), nbas: 1)
        >>> node3.basis_set.nbas
        1
        """
        super().__init__()
        if isinstance(basis_sets, BasisSet):
            basis_sets = [basis_sets]
        elif basis_sets is None or len(basis_sets) == 0:
            global DUMMY_IDX
            basis_sets = [BasisDummy(("Virtual DOF", DUMMY_IDX))]
            DUMMY_IDX += 1
        self.basis_sets: List[BasisSet] = basis_sets
        self.n_sets = len(basis_sets)
        qn_size_list = [b.sigmaqn.shape[1] for b in self.basis_sets]
        if len(set(qn_size_list)) != 1:
            raise ValueError(f"Inconsistent quantum number size: {set(qn_size_list)}")
        self.qn_size: int = qn_size_list[0]
        self.dofs = [b.dofs for b in basis_sets]
        self.pbond_dims = [len(b.sigmaqn) for b in self.basis_sets]

    def copy(self):
        new = self.__class__(self.basis_sets)
        if self.parent is not None:
            new.parent = self.parent.copy()
        new.children = self.children.copy()
        return new

    @property
    def basis_set(self):
        if len(self.basis_sets) != 1:
            raise ValueError("This node has multiple basis sets. Use self.basis_sets[0] instead.")
        return self.basis_sets[0]

    def __str__(self):
        if len(self.basis_sets) == 1:
            content = str(self.basis_set)
        else:
            content = ", ".join(str(b) for b in self.basis_sets)
        return f"{self.__class__.__name__}({content})"

    def __repr__(self):
        return str(self)

class TreeNodeTensor(TreeNode):
    def __init__(self, tensor, qn=None):
        """
        Tree node whose data is numerical tensors for each TTN node/site.
        The indices of the tensor are ordered as follows:
        [child1, child2, ..., childN, physical1, physical2, ..., physicalN, parent]

        Parameters
        ----------
        tensor: The numerical tensor
        qn: The quantum number from the tensor to its parent.
        """
        super().__init__()
        self.tensor: np.ndarray = tensor
        self.qn: np.ndarray = qn

    def check_canonical(self, atol=None, assertion=True):
        if atol is None:
            atol = backend.canonical_atol
        tensor = self.tensor.reshape(-1, self.tensor.shape[-1])
        s = tensor.conj().T @ tensor
        res = np.allclose(s, np.eye(s.shape[0]), atol=atol)
        if assertion:
            assert res
        return res

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, tensor):
        if np.iscomplexobj(tensor):
            dtype = backend.complex_dtype
        else:
            dtype = backend.real_dtype
        self._tensor = np.asarray(asnumpy(tensor), dtype=dtype)

    # alias
    array = tensor

    @property
    def qn(self):
        return self._qn

    @qn.setter
    def qn(self, qn):
        self._qn = np.array(qn)

    def __str__(self):
        content = str(self.shape) + "," + str(self.tensor.dtype)
        return f"{self.__class__.__name__}({content})"

    def __repr__(self):
        return str(self)


class TreeNodeEnviron(TreeNode):
    # tree node whose data is environmental tensors
    def __init__(self):
        super().__init__()
        self.parent: TreeNodeEnviron = None
        # environ from parent
        self.environ_parent: np.ndarray = None
        # environ from children
        self.environ_children: List[np.ndarray] = []


NodeUnion = Union[TreeNodeTensor, TreeNodeBasis, TreeNodeEnviron]


def copy_connection(source_node_list: List[NodeUnion], target_node_list: List[NodeUnion]) -> NodeUnion:
    """
    Create a new tree with the same connection structure as the source tree in the target tree.

    Parameters
    ----------
    source_node_list : List[NodeUnion]
        The list of nodes in the source tree.
    target_node_list : List[NodeUnion]
        The list of nodes in the target tree.

    Returns
    -------
    NodeUnion
        The root node of the target tree.
    """
    node2idx: Dict[NodeUnion, int] = {n: i for i, n in enumerate(source_node_list)}
    root = None
    for source_node, target_node in zip(source_node_list, target_node_list):
        for child in source_node.children:
            idx = node2idx[child]
            target_node.add_child(target_node_list[idx])
        if source_node.parent is None:
            root = target_node
    assert root is not None
    return root
