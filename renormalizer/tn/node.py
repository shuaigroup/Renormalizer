from typing import List, Dict, Any, Union

from renormalizer.model.basis import BasisSet, BasisHalfSpin
from renormalizer.mps.backend import xp


class TreeNode:
    def __init__(self):
        self.children: List[__class__] = []
        self.parent: TreeNode = None
        self.workspace: Any = None

    def add_child(self, node: "TreeNode"):
        assert node.parent is None
        self.children.append(node)
        node.parent = self


class TreeNodeBasis(TreeNode):
    def __init__(self, basis_set):
        super().__init__()
        self.basis_set: BasisSet = basis_set


class TreeNodeTensor(TreeNode):
    def __init__(self, tensor):
        super().__init__()
        self.tensor: xp.ndarray = tensor


class TreeNodeEnviron(TreeNode):
    def __init__(self):
        super().__init__()
        self.parent: TreeNodeEnviron = None
        # environ from parent
        self.environ_parent: xp.ndarray = None
        # environ from children
        self.environ_children: List[xp.ndarray] = []


NodeUnion = Union[TreeNodeTensor, TreeNodeBasis, TreeNodeEnviron]


def copy_connection(source_node_list: List[NodeUnion], target_node_list: List[NodeUnion]) -> NodeUnion:
    node2idx: Dict[NodeUnion, int] = {n:i for i, n in enumerate(source_node_list)}
    root = None
    for source_node, target_node in zip(source_node_list, target_node_list):
        for child in source_node.children:
            idx = node2idx[child]
            target_node.add_child(target_node_list[idx])
        if source_node.parent is None:
            root = target_node
    assert root is not None
    return root


