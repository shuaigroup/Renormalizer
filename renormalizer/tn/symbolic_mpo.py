import logging
from typing import List

import numpy as np

from renormalizer import Op, Model
from renormalizer.tn.tree import BasisTree
from renormalizer.mps.symbolic_mpo import _terms_to_table, _transform_table, _construct_symbolic_mpo_one_site, OpTuple


logger = logging.getLogger(__name__)


# translate the numbers into symbolic Matrix Operator
def compose_symbolic_mo_general(in_ops_list, out_ops, primary_ops):
    shape = [len(in_ops) for in_ops in in_ops_list] + [len(out_ops)]
    mo = np.full(shape, None, dtype=object)
    for i, _ in np.ndenumerate(mo):
        mo[i] = []
    for iop, out_op in enumerate(out_ops):
        for composed_op in out_op:
            op = primary_ops[composed_op.symbol[-1]]
            if in_ops_list:
                in_idx = tuple(composed_op.symbol[:-1])
                l = mo[in_idx][iop]
            else:
                l = mo[iop]
            l.append(composed_op.factor * op)
    return mo



def construct_symbolic_mpo(tn:BasisTree, terms: List[Op], const:float=0):
    algo = "Hopcroft-Karp"
    nodes = tn.postorder_list()
    basis = [n.basis_set for n in nodes]
    model = Model(basis, [])
    qn_size = model.qn_size
    table, factor = _terms_to_table(model, terms, const)
    table, factor, primary_ops = _transform_table(table, factor)

    dummy_in_ops = [[OpTuple([0], qn=np.zeros(qn_size, dtype=int), factor=1)]]
    out_ops_list = []

    for i, node in enumerate(nodes):
        if not node.children:
            ta = np.zeros((table.shape[0], 1), dtype=np.uint16)
            table = np.concatenate((ta, table), axis=1)
            table_row = table[:, :2]
            table_col = table[:, 2:]
            in_ops_list = [dummy_in_ops]
        else:
            # the children must have been visited
            children_idx = [nodes.index(n) for n in node.children]
            assert np.all(np.array(children_idx) < i)
            in_ops_list = [out_ops_list[i] for i in children_idx]
            m = len(node.children)
            # roll relevant columns to the front
            table = np.roll(table, m, axis=1)
            table_row = table[:, :m+1]
            table_col = table[:, m+1:]
        out_ops, table, factor = \
            _construct_symbolic_mpo_one_site(table_row, table_col, in_ops_list, factor, primary_ops, algo)
        # move the new column at the first index to the last index
        table = np.roll(table, -1, axis=1)
        out_ops_list.append(out_ops)

    mpo = []
    for i, node in enumerate(nodes):
        children_idx = [nodes.index(n) for n in node.children]
        in_ops_list = [out_ops_list[i] for i in children_idx]
        mo = compose_symbolic_mo_general(in_ops_list, out_ops_list[i], primary_ops)
        mpo.append(mo)

    return mpo
