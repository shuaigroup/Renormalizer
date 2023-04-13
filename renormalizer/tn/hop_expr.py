import opt_einsum as oe

from renormalizer.mps.backend import np
from renormalizer.mps.matrix import asxp
from renormalizer.tn.node import TreeNodeTensor
from renormalizer.tn.tree import TTNS, TTNO, TTNEnviron


def hop_expr1(snode: TreeNodeTensor, ttns: TTNS, ttno: TTNO, ttne: TTNEnviron):
    # one site
    enode = ttne.node_list[ttns.node_idx[snode]]
    onode = ttno.node_list[ttns.node_idx[snode]]

    args = []
    # enode children environments
    for i, echild_environchild in enumerate(enode.environ_children):
        args.append(echild_environchild)
        args.append(ttne.get_child_indices(enode, i, ttns, ttno))
    # parent environments
    args.append(enode.environ_parent)
    args.append(ttne.get_parent_indices(enode, ttns, ttno))
    # operator
    args.extend([onode.tensor, ttno.get_node_indices(onode, "up", "down")])

    # input and output
    input_indices = ttns.get_node_indices(snode)
    output_indices = ttns.get_node_indices(snode, True)

    shape = snode.shape
    # cache the contraction path
    expr = _contract_expression(args, shape, input_indices, output_indices)
    hdiag = _get_hdiag(args, input_indices)
    return expr, hdiag


def hop_expr2(snode: TreeNodeTensor, ttns: TTNS, ttno: TTNO, ttne: TTNEnviron):
    # two sites
    sparent = snode.parent
    enode = ttne.node_list[ttns.node_idx[snode]]
    eparent = ttne.node_list[ttns.node_idx[sparent]]
    onode = ttno.node_list[ttns.node_idx[snode]]
    oparent = ttno.node_list[ttns.node_idx[sparent]]

    args = []
    # enode children environments
    for i, echild_environchild in enumerate(enode.environ_children):
        args.append(echild_environchild)
        args.append(ttne.get_child_indices(enode, i, ttns, ttno))

    # eparent children environments
    for i, enode_environchild in enumerate(eparent.environ_children):
        if eparent.children[i] is enode:
            continue
        args.append(enode_environchild)
        args.append(ttne.get_child_indices(eparent, i, ttns, ttno))

    # eparent parent environments
    args.append(eparent.environ_parent)
    args.append(ttne.get_parent_indices(eparent, ttns, ttno))

    # operator
    args.extend([oparent.tensor, ttno.get_node_indices(oparent, "up", "down")])
    args.extend([onode.tensor, ttno.get_node_indices(onode, "up", "down")])

    # input and output
    input_indices = ttns.get_node_indices(snode, include_parent=True)
    output_indices = ttns.get_node_indices(snode, True, include_parent=True)

    # shape
    shape = list(snode.shape[:-1])
    shape_parent = list(snode.parent.shape)
    del shape_parent[snode.parent.children.index(snode)]
    shape += shape_parent
    # cache the contraction path
    expr = _contract_expression(args, shape, input_indices, output_indices)
    hdiag = _get_hdiag(args, input_indices)
    return expr, hdiag


def _contract_expression(args, x_shape, x_indices, y_indices):
    # contract_expression in interleaved format
    args_fake = args.copy()
    args_fake.extend([np.empty(x_shape), x_indices])
    args_fake.append(y_indices)
    indices, tensors = oe.parser.convert_interleaved_input(args_fake)
    args = [asxp(t) for t in tensors[:-1]] + [x_shape]
    expr = oe.contract_expression(
        indices,
        *args,
        constants=list(range(len(tensors)))[:-1],
    )
    return expr


def _get_hdiag(args, input_indices):
    new_args = []
    for arg in args:
        if not isinstance(arg, (tuple, list)):
            # tensors
            new_args.append(asxp(arg))
            continue
        # indices
        arg = list(arg)
        if arg[0][-5:] == "_conj":
            # the environ
            arg[0] = arg[0][:-5]
        elif arg[1] == "up":
            # mpo part
            arg[1] = "down"
        else:
            pass
        new_args.append(tuple(arg))
    new_args.append(input_indices)
    return oe.contract(*new_args)
