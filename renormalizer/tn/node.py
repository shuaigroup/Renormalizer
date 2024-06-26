from typing import List, Dict, Union, Sequence

from renormalizer.mps.backend import np, backend
from renormalizer.mps.matrix import asnumpy
from renormalizer.model.basis import BasisSet


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


class TreeNodeBasis(TreeNode):
    # tree node whose data is basis sets
    def __init__(self, basis_sets: List[BasisSet]):
        super().__init__()
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

    @property
    def qn(self):
        return self._qn

    @qn.setter
    def qn(self, qn):
        self._qn = np.array(qn)


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
