from itertools import chain
import logging
from typing import List

from renormalizer.mps.backend import np
from renormalizer import Op, Model
from renormalizer.model.basis import BasisSet
from renormalizer.tn.treebase import BasisTree
from renormalizer.mps.symbolic_mpo import _terms_to_table, _transform_table, _construct_symbolic_mpo_one_site, OpTuple


logger = logging.getLogger(__name__)


# translate the numbers into symbolic Matrix Operator
def compose_symbolic_mo_general(in_ops_list, out_ops, primary_ops, k):
    shape = [len(in_ops) for in_ops in in_ops_list] + [len(out_ops)]
    mo = np.full(shape, None, dtype=object)
    for i, _ in np.ndenumerate(mo):
        mo[i] = []
    for iop, out_op in enumerate(out_ops):
        for composed_op in out_op:
            if in_ops_list:
                in_idx = tuple(composed_op.symbol[:-k])
                l = mo[in_idx][iop]
            else:
                l = mo[iop]
            op = composed_op.factor
            for s in composed_op.symbol[-k:]:
                op = op * primary_ops[s]
            l.append(op)
    return mo


# translate symbolic Matrix Operator to numerical matrix operator defined with certain basis
def symbolic_mo_to_numeric_mo_general(basis_sets: List[BasisSet], mo, dtype):
    model = Model(basis_sets, [])
    pdims = [b.nbas for b in basis_sets]
    shape = list(mo.shape) + list(chain(*[[pdim, pdim] for pdim in pdims]))
    mo_tensor = np.zeros(shape, dtype=dtype)
    terms: List[Op]
    for i, terms in np.ndenumerate(mo):
        for term in terms:
            term_split, factor = term.split_elementary(model.dof_to_siteidx)
            assert len(term_split) == len(basis_sets)
            mo_elem = np.eye(1) * factor
            for symbol, b in zip(term_split, basis_sets):
                mo_elem = np.tensordot(mo_elem, b.op_mat(symbol)[None, :, :, None], axes=1)
            assert not np.iscomplexobj(mo_elem), "complex operator not supported yet"
            mo_tensor[i] += mo_elem[0, ..., 0]

    return np.moveaxis(mo_tensor, mo.ndim - 1, -1)


def construct_symbolic_mpo(tn: BasisTree, terms: List[Op], const: float = 0):
    algo = "Hopcroft-Karp"
    nodes = tn.postorder_list()
    basis = list(chain(*[n.basis_sets for n in nodes]))
    model = Model(basis, [])
    qn_size = model.qn_size
    table, factor = _terms_to_table(model, terms, const)
    table, factor, primary_ops = _transform_table(table, factor)

    dummy_in_ops = [[OpTuple([0], qn=np.zeros(qn_size, dtype=int), factor=1)]]
    out_ops: List[List[OpTuple]]
    out_ops_list = []

    for i, node in enumerate(nodes):
        k = node.n_sets
        if not node.children:
            ta = np.zeros((table.shape[0], 1), dtype=np.uint16)
            table = np.concatenate((ta, table), axis=1)
            table_row = table[:, : k + 1]
            table_col = table[:, k + 1 :]
            in_ops_list = [dummy_in_ops]
        else:
            # the children must have been visited
            children_idx = [nodes.index(n) for n in node.children]
            assert np.all(np.array(children_idx) < i)
            in_ops_list = [out_ops_list[i] for i in children_idx]
            m = len(node.children)
            # roll relevant columns to the front
            table = np.roll(table, m, axis=1)
            table_row = table[:, : m + k]
            table_col = table[:, m + k :]
        out_ops, table, factor = _construct_symbolic_mpo_one_site(
            table_row, table_col, in_ops_list, factor, primary_ops, algo, k
        )
        # move the new column at the first index to the last index
        table = np.roll(table, -1, axis=1)
        out_ops_list.append(out_ops)

    mpo = []
    for i, node in enumerate(nodes):
        children_idx = [nodes.index(n) for n in node.children]
        in_ops_list = [out_ops_list[i] for i in children_idx]
        mo = compose_symbolic_mo_general(in_ops_list, out_ops_list[i], primary_ops, node.n_sets)
        mpo.append(mo)

    mpoqn = []
    for out_ops in out_ops_list:
        qn = np.array([out_op[0].qn for out_op in out_ops])
        mpoqn.append(qn)

    return mpo, mpoqn
