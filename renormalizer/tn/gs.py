import logging
from typing import List

import numpy as np
import scipy
import opt_einsum as oe

from renormalizer.lib import davidson
from renormalizer.mps.backend import primme, IMPORT_PRIMME_EXCEPTION
from renormalizer.mps.svd_qn import get_qn_mask
from renormalizer.tn.node import TreeNodeTensor
from renormalizer.tn.tree import TensorTreeState, TensorTreeOperator, TensorTreeEnviron


logger = logging.getLogger(__name__)


def optimize_tts(tts: TensorTreeState, tto: TensorTreeOperator, procedure):
    tte = TensorTreeEnviron(tts, tto)
    e_list = []
    for m, percent in procedure:
        # todo: better converge condition
        micro_e = optimize_recursion(tts.root, tts, tto, tte, m, percent)
        logger.info(f"Micro e: {micro_e}")
        e_list.append(micro_e[-1])
    return e_list


def optimize_recursion(snode: TreeNodeTensor, tts, tto, tte, m:int, percent:float=0) -> List[float]:
    micro_e = []
    for ichild, child in enumerate(snode.children):

        if child.children:
            # optimize snode + child
            e, c = optimize_2site(child, tts, tto, tte)
            micro_e.append(e)
            # cano to child
            tts.update_2site(child, c, m, percent, cano_parent=False)
            # update env
            tte.update_2site(child, tts, tto)
            # recursion optimization
            micro_e_child = optimize_recursion(child, tts, tto, tte, m)
            micro_e.extend(micro_e_child)

        # optimize snode + child
        e, c = optimize_2site(child, tts, tto, tte)
        micro_e.append(e)
        # cano to snode
        tts.update_2site(child, c, m, percent, cano_parent=True)
        # update env
        tte.update_2site(child, tts, tto)
    return micro_e


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
    args.extend([oparent.tensor, tto.get_node_indices(oparent, "up", "down")])
    args.extend([onode.tensor, tto.get_node_indices(onode, "up", "down")])

    # input and output
    cguess, input_indices = merge_parent(snode, tts)
    qnmat = tts.get_qnmat(snode)[-1]
    qn_mask = get_qn_mask(qnmat, tts.qntot)
    shape = cguess.shape
    assert qn_mask.shape == shape
    cguess = cguess[qn_mask].ravel()
    h_dim = len(cguess)
    output_indices = tts.get_node_indices(snode, True)
    shared_bond = output_indices[-1]
    output_indices.extend(tts.get_node_indices(snode.parent, True))
    for i in range(2):
        output_indices.remove(shared_bond)

    hdiag = _get_hdiag(args, input_indices)[qn_mask].ravel()
    assert len(hdiag) == h_dim
    # cache the contraction path
    expr = hop_expr(args, shape, input_indices, output_indices)
    def hop(x):
        cstruct = vec2tensor(x, qn_mask)
        return expr(cstruct)[qn_mask].ravel()

    # todo: choose solver in the future
    if True:
        precond = lambda x, e, *args: x / (hdiag - e + 1e-4)

        e, c = davidson(
            hop, cguess, precond, max_cycle=100, nroots=1, max_memory=64000, verbose=0
        )
    elif True:
        if primme is None:
            logger.error("can not import primme")
            raise IMPORT_PRIMME_EXCEPTION
        precond = lambda x: scipy.sparse.diags(1 / (hdiag + 1e-4)) @ x
        A = scipy.sparse.linalg.LinearOperator((h_dim, h_dim), matvec=hop, matmat=hop)
        M = scipy.sparse.linalg.LinearOperator((h_dim, h_dim), matvec=precond, matmat=hop)
        e, c = primme.eigsh(
            A,
            k=1,
            which="SA",
            v0=np.array(cguess).reshape(-1, 1),
            OPinv=M,
            method="PRIMME_DYNAMIC",
            tol=1e-6,
        )
        c = c[:, 0]
        e = e[0]
    elif True:
        A = scipy.sparse.linalg.LinearOperator((h_dim,h_dim), matvec=hop)
        e, c = scipy.sparse.linalg.eigsh(A, k=1, which="SA", v0=cguess)
        e = e[0]
    else:
        # direct algorithm. Poor performance, debugging only.
        a_list = []
        for i in range(h_dim):
            a = np.zeros(h_dim)
            a[i] = 1
            a_list.append(hop(a))
        a = np.array(a_list)
        assert np.allclose(a, a.conj().T)
        evals, evecs = np.linalg.eigh(a)
        e = evals[0]
        c = evecs[:, 0]
    c = vec2tensor(c, qn_mask)
    return e, c


def merge_parent(snode, tts: TensorTreeState):
    # merge a node with its parent
    args = []
    snode_indices = tts.get_node_indices(snode)
    parent_indices = tts.get_node_indices(snode.parent)
    args.extend([snode.tensor, snode_indices])
    args.extend([snode.parent.tensor, parent_indices])
    output_indices = snode_indices + parent_indices
    shared_bond = snode_indices[-1]
    for i in range(2):
        output_indices.remove(shared_bond)
    args.append(output_indices)
    return oe.contract(*args), output_indices


def _get_hdiag(args, input_indices):
    new_args = []
    for arg in args:
        if not isinstance(arg, tuple):
            new_args.append(arg)
            continue
        arg = list(arg)
        if arg[0][-5:] == "_conj":
            # the environ
            arg[0] = arg[0][:-5]
        elif arg[1] == "up":
            # mpo part
            arg[1] = "down"
        else:
            pass
        args.append(tuple(arg))
    new_args.append(input_indices)
    return oe.contract(*new_args)


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


def vec2tensor(c, qn_mask):
    cstruct = np.zeros(qn_mask.shape, dtype=c.dtype)
    np.place(cstruct, qn_mask, c)
    return cstruct
