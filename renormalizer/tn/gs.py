import numpy as np
import scipy
import opt_einsum as oe

from renormalizer.tn.node import TreeNodeTensor
from renormalizer.tn.tree import TensorTreeState, TensorTreeOperator, TensorTreeEnviron


def optimize_tts(tts: TensorTreeState, tto: TensorTreeOperator, m:int):
    tte = TensorTreeEnviron(tts, tto)
    for i in range(20):
        e = optimize_recursion(tts.root, tts, tto, tte, m)
    return e


def optimize_recursion(snode: TreeNodeTensor, tts, tto, tte, m:int):
    for ichild, child in enumerate(snode.children):

        if child.children:
            # optimize snode + child
            e, c = optimize_2site(child, tts, tto, tte)
            # cano to child
            tts.update_2site(child, c, m, cano_parent=False)
            # update env
            tte.update_2site(child, tts, tto)
            # recursion optimization
            optimize_recursion(child, tts, tto, tte, m)

        # optimize snode + child
        e, c = optimize_2site(child, tts, tto, tte)
        # cano to snode
        tts.update_2site(child, c, m, cano_parent=True)
        # update env
        tte.update_2site(child, tts, tto)
    return e


def optimize_2site(snode: TreeNodeTensor, tts: TensorTreeState, tto: TensorTreeOperator, tte: TensorTreeEnviron):
    sparent = snode.parent
    enode = tte.node_list[tts.node_idx[snode]]
    eparent = tte.node_list[tts.node_idx[sparent]]
    onode = tto.node_list[tts.node_idx[snode]]
    oparent = tto.node_list[tts.node_idx[sparent]]

    args = []
    # enode children environments
    for i, echild_environchild in enumerate(enode.environ_children):
        args.append(echild_environchild)
        args.append(tte.get_child_indices(enode, i, tts, tto))

    # eparent children environments
    for i, enode_environchild in enumerate(eparent.environ_children):
        if eparent.children[i] is enode:
            continue
        args.append(enode_environchild)
        args.append(tte.get_child_indices(eparent, i, tts, tto))

    # eparent parent environments
    args.append(eparent.environ_parent)
    args.append(tte.get_parent_indices(eparent, tts, tto))

    # operator
    args.extend([oparent.tensor, tto.get_node_indices(oparent, "bra", "ket")])
    args.extend([onode.tensor, tto.get_node_indices(onode, "bra", "ket")])

    # input and output
    cguess, merged_indices = merge_parent(snode, tts)
    shape = cguess.shape
    cguess = cguess.ravel()
    dim = cguess.shape[0]
    output_indices = tts.get_node_indices(snode, "bra", True)
    shared_bond = output_indices[-1]
    output_indices.extend(tts.get_node_indices(snode.parent, "bra", True))
    for i in range(2):
        output_indices.remove(shared_bond)

    # cache the contraction path
    expr = hop_expr(args, shape, merged_indices, output_indices)
    def hop(x):
        return expr(x.reshape(shape)).ravel()

    # todo: add preconditioner
    A = scipy.sparse.linalg.LinearOperator((dim,dim), matvec=hop)
    e, c = scipy.sparse.linalg.eigsh(A, k=1, which="SA", v0=cguess)
    e = e[0]
    return e, c.reshape(shape)


def merge_parent(snode, tts: TensorTreeState):
    # merge a node with its parent
    args = []
    snode_indices = tts.get_node_indices(snode, "ket")
    parent_indices = tts.get_node_indices(snode.parent, "ket")
    args.extend([snode.tensor, snode_indices])
    args.extend([snode.parent.tensor, parent_indices])
    output_indices = snode_indices + parent_indices
    shared_bond = snode_indices[-1]
    for i in range(2):
        output_indices.remove(shared_bond)
    args.append(output_indices)
    return oe.contract(*args), output_indices


def hop_expr(args, x_shape, x_indices, y_indices):
    args_fake = args.copy()
    args_fake.extend([np.empty(x_shape), x_indices])
    args_fake.append(y_indices)
    indices, tensors = oe.parser.convert_interleaved_input(args_fake)
    args = tensors[:-1] + [x_shape]
    expr = oe.contract_expression(
        indices,
        *args,
        constants=list(range(len(tensors)))[:-1],
    )
    return expr
