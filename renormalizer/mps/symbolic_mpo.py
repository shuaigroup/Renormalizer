# -*- coding: utf-8 -*-
import logging
from collections import namedtuple, OrderedDict, defaultdict
from typing import List, Set, Tuple, Dict

import numpy as np
import scipy
import scipy.sparse

from renormalizer.model import Model
from renormalizer.model.basis import BasisSet
from renormalizer.model.op import Op
from renormalizer.lib import bipartite_vertex_cover

logger = logging.getLogger(__name__)

# The `Op` class is transformed to a light-weight named tuple
# for better performance
OpTuple = namedtuple("OpTuple", ["symbol", "qn", "factor"])


def construct_symbolic_mpo(table, factor, algo="Hopcroft-Karp"):
    r"""
    A General Compact (Symbolic) MPO Construction Routine

    Args:

    table: an operator table with shape (operator nterm, nsite). Each entry contains elementary operators on each site.
    factor (np.ndarray): one prefactor vector (dim: operator nterm)
    algo: the algorithm used to select local ops, "Hopcroft-Karp"(default), "Hungarian".
          They are both global optimal and have only minor performance difference.

    Note:
    op with the same op.symbol must have the same op.qn and op.factor

    Return:
    mpo: symbolic mpo
    mpoqn: quantum number
    qntot: total quantum number of the operator
    qnidx: the index of the qn

    The idea:

    the index of the primary ops {0:"I", 1:"a", 2:r"a^\dagger"}

    for example: H = 2.0 * a_1 a_2^dagger   + 3.0 * a_2^\dagger a_3 + 4.0*a_0^\dagger a_3
    The column names are the site indices with 0 and 4 imaginary (see the note below)
    and the content of the table is the index of primary operators.
                        s0   s1   s2   s3  s4  factor
    a_1 a_2^dagger      0    1    2    0   0   2.0
    a_2^\dagger a_3     0    0    2    1   0   3.0
    a_1^\dagger a_3     0    2    0    1   0   4.0
    for convenience the first and last column mean that the operator of the left and right hand of the system is I

    cut the string to construct the row(left) and column(right) operator and find the duplicated/independent terms
                        s0   s1 |  s2   s3  s4 factor
    a_1 a_2^dagger      0    1  |  2    0   0  2.0
    a_2^\dagger a_3     0    0  |  2    1   0  3.0
    a_1^\dagger a_3     0    2  |  0    1   0  4.0

     The content of the table below means matrix elements with basis explained in the notes.
     In the matrix elements, 1 means combination and 0 means no combination.
          (2,0,0) (2,1,0) (0,1,0)  -> right side of the above table
     (0,1)   1       0       0
     (0,0)   0       1       0
     (0,2)   0       0       1
       |
       v
     left side of the above table
     In this case all operators are independent so the content of the matrix is diagonal

    and select the terms and rearrange the table
    The selection rule is to find the minimal number of rows+cols that can eliminate the
    matrix
                      s1   s2 |  s3 s4 factor
    a_1 a_2^dagger    0'   2  |  0  0  2.0
    a_2^\dagger a_3   1'   2  |  1  0  3.0
    a_1^\dagger a_3   2'   0  |  1  0  4.0
    0'/1'/2' are three new operators(could be non-elementary)
    The local mpo is the transformation matrix between 0,0,0 to 0',1',2'.
    In this case, the local mpo is simply (1, 0, 2)

    cut the string and find the duplicated/independent terms
            (0,0), (1,0)
     (0',2)   1      0
     (1',2)   0      1
     (2',0)   0      1

    and select the terms and rearrange the table
    apparently choose the (1,0) column and construct the complementary operator (1',2)+(2',0) is better
    0'' =  3.0 * (1', 2) + 4.0 * (2', 0)
                                                 s2     s3 | s4 factor
    (4.0 * a_1^dagger + 3.0 * a_2^dagger) a_3    0''    1  | 0  1.0
    a_1 a_2^dagger                               1''    0  | 0  2.0
    0''/1'' are another two new operators(non-elementary)
    The local mpo is the transformation matrix between 0',1',2' to 0'',1''

             (0)
     (0'',1)  1
     (1'',0)  1

    The local mpo is the transformation matrix between 0'',1'' to 0'''
    """

    qn_size = len(table[0][0].qn)
    # Simplest case. Cut to the chase
    if len(table) == 1:
        # The first layer: number of sites. The middle array: in and out virtual bond
        # the 4th layer: operator sums
        mpo: List[np.ndarray[List[Op]]] = []
        mpoqn = [np.zeros((1, qn_size), dtype=int)]
        primary_ops = list(set(table[0]))
        op2idx = dict(zip(primary_ops, range(len(primary_ops))))
        out_ops_list: List[List[OpTuple]] = [[OpTuple([0], qn=0, factor=1)]]
        for op in table[0]:
            mo = np.full((1, 1), None)
            mo[0][0] = [op]
            mpo.append(mo)
            qn = mpoqn[-1][0] + op.qn
            mpoqn.append(np.array([qn]))
            out_ops_list.append([OpTuple([0, op2idx[op]], qn=qn, factor=1)])

        mpo[-1][0][0][0] = factor[0] * mpo[-1][0][0][0]
        last_optuple = out_ops_list[-1][0]
        out_ops_list[-1][0] = OpTuple(last_optuple.symbol, qn=last_optuple.qn, factor=factor[0]*last_optuple.factor)
        qntot = qn
        mpoqn[-1] = np.zeros((1, qn_size), dtype=int)
        qnidx = len(mpo) - 1
        # the last two terms are not set for fast construction of the operators
        return mpo, mpoqn, qntot, qnidx, out_ops_list, primary_ops

    logger.debug(f"symbolic mpo algorithm: {algo}")
    logger.debug(f"Input operator terms: {len(table)}")

    table, factor, primary_ops = _transform_table(table, factor)

    # add the first and last column for convenience
    ta = np.zeros((table.shape[0], 1), dtype=np.uint16)
    table = np.concatenate((ta, table, ta), axis=1)
    logger.debug(f"After combination of the same terms: {table.shape[0]}")

    # 0 represents the identity symbol. Identity might not present
    # in `primary_ops` but the algorithm still works.

    in_ops = [[OpTuple([0], qn=np.zeros(qn_size, dtype=int), factor=1)]]

    out_ops_list = _construct_symbolic_mpo(table, in_ops, factor, primary_ops, algo)
    # number of sites + 1. Note that the table was expanded for convenience
    assert len(out_ops_list) == len(table[0]) - 1
    mpo = []
    for i in range(len(out_ops_list)-1):
        mo = compose_symbolic_mo(out_ops_list[i], out_ops_list[i+1], primary_ops)
        mpo.append(mo)

    mpoqn = []
    for out_ops in out_ops_list:
        qn = np.array([out_op[0].qn for out_op in out_ops])
        mpoqn.append(qn)

    qntot = mpoqn[-1][0]
    mpoqn[-1] = np.zeros((1, qn_size), dtype=int)
    qnidx = len(mpo) - 1

    return mpo, mpoqn, qntot, qnidx, out_ops_list, primary_ops


def _construct_symbolic_mpo(table, in_ops, factor, primary_ops, algo="qr"):
    assert len(np.unique(table, axis=0)) == len(table)
    nsite = table.shape[1] - 2

    out_ops_list = [in_ops]

    for isite in range(nsite):
        table_row = table[:, :2]
        table_col = table[:, 2:]
        out_ops, table, factor = _construct_symbolic_mpo_one_site(table_row, table_col, [in_ops], factor, primary_ops, algo)

        # debug
        # logger.debug(f"in_ops: {in_ops}")
        # logger.debug(f"out_ops: {out_ops}")
        # logger.debug(f"new_factor: {factor}")

        in_ops = out_ops
        out_ops_list.append(out_ops)


    assert len(factor) == 1 and len(table) == 1
    assert factor[0] == 1
    return out_ops_list


def _construct_symbolic_mpo_one_site(table_row, table_col, in_ops_list, factor, primary_ops, algo, k=1):
    # split table into the row and col part
    term_row, row_unique_inverse = np.unique(table_row, axis=0, return_inverse=True)
    assert len(in_ops_list) + k == term_row.shape[1]

    # faster version of the following code
    # term_col, col_unique_inverse = np.unique(table_col, axis=0, return_inverse=True)
    term_col = {}
    col_unique_inverse = []
    for row in table_col:
        row_bytes = row.tobytes()
        i_and_row = term_col.get(row_bytes)
        if i_and_row is None:
            i_and_row = (len(term_col), row)
            term_col[row_bytes] = i_and_row
        col_unique_inverse.append(i_and_row[0])
    term_col = [v[1] for v in term_col.values()]

    non_red = scipy.sparse.coo_matrix((np.arange(len(factor)) + 1, (row_unique_inverse, col_unique_inverse))).tocsr()

    if not algo.startswith("qr"):
        return _decompose_graph(term_row, term_col, non_red, in_ops_list, factor, primary_ops, algo, k)
    else:
        return _decompose_qr(term_row, term_col, non_red, in_ops_list, factor, primary_ops, algo, k)



def _decompose_graph(term_row, term_col, non_red, in_ops_list, factor, primary_ops, algo, k=1):
    bigraph = []
    if non_red.shape[0] < non_red.shape[1]:
        for i in range(non_red.shape[0]):
            bigraph.append(non_red.indices[non_red.indptr[i]:non_red.indptr[i + 1]])
        rowbool, colbool = bipartite_vertex_cover(bigraph, algo=algo)
    else:
        non_red_csc = non_red.tocsc()
        for i in range(non_red.shape[1]):
            bigraph.append(non_red_csc.indices[non_red_csc.indptr[i]:non_red_csc.indptr[i + 1]])
        colbool, rowbool = bipartite_vertex_cover(bigraph, algo=algo)

    row_select = np.nonzero(rowbool)[0]
    # largest cover first
    row_select = sorted(row_select, key=lambda i: non_red.indptr[i + 1] - non_red.indptr[i], reverse=True)
    col_select = np.nonzero(colbool)[0]

    if len(row_select) > 0:
        assert np.amax(row_select) < non_red.shape[0]
    if len(col_select) > 0:
        assert np.amax(col_select) < non_red.shape[1]

    # select the reserved ops
    out_ops: List[List[OpTuple]] = []
    new_table = []
    new_factor = []

    for row_idx in row_select:
        # construct out_op
        # dealing with row (left side of the table). One row corresponds to multiple cols.
        # Produce one out operator and multiple new_table entries
        symbol = term_row[row_idx]
        qn = _compute_qn(in_ops_list, symbol, primary_ops, k)
        out_op = OpTuple(symbol, qn, factor=1.0)
        out_ops.append([out_op])

        col_link = non_red.indices[non_red.indptr[row_idx]:non_red.indptr[row_idx + 1]]
        stack = np.array([len(out_ops) - 1] * len(col_link), dtype=np.uint16).reshape(-1, 1)
        new_table.append(np.hstack((stack, [term_col[i] for i in col_link])))
        new_factor.append(factor[non_red[row_idx, col_link].toarray() - 1])
        non_red.data[non_red.indptr[row_idx]:non_red.indptr[row_idx + 1]] = 0

    non_red.eliminate_zeros()

    nonzero_row_idx, nonzero_col_idx = non_red.nonzero()
    for col_idx in col_select:

        out_ops.append([])
        # complementary operator
        # dealing with column (right side of the table). One col correspond to multiple rows.
        # Produce multiple out operators and one new_table entry
        non_red_one_col = non_red[:, col_idx].toarray().flatten()
        for i in nonzero_row_idx[np.nonzero(nonzero_col_idx == col_idx)[0]]:
            symbol = term_row[i]
            qn = _compute_qn(in_ops_list, symbol, primary_ops, k)
            out_op = OpTuple(symbol, qn, factor=factor[non_red_one_col[i] - 1])
            out_ops[-1].append(out_op)

        new_table.append(np.array([len(out_ops) - 1] + list(term_col[col_idx]), dtype=np.uint16).reshape(1, -1))
        new_factor.append(1.0)

        # it is not necessary to remove the column nonzero elements
        # non_red[:, col_idx] = 0
        # non_red.eliminate_zeros()

    # reconstruct the table in new operator
    table = np.concatenate(new_table)
    # check the number of incoming operators could be represented as np.uint16
    assert len(out_ops) <= np.iinfo(np.uint16).max
    factor = np.concatenate(new_factor, axis=None)

    assert len(table) == len(factor)
    return out_ops, table, factor


def _decompose_qr(term_row, term_col, non_red, in_ops_list, factor, primary_ops, algo, k=1):
    r"""
    The overall operator is written as

    \hat{O} = \sum_{jk}\Gamma_{jk} \hat L_j \otimes \hat R_k

    where $\hat L_j$ and $\hat R_k$ are unique operators from the left side and the right side respectively.
    $\hat L$ is represented by `term_row` and $\hat R$ is represented by `term_col`.

    Then decompose $\Gamma_{jk} = \sum_l Q_{jl}R_{lk}$ with QR decomposition and the operator becomes

    \hat O =  \sum_l (\sum_{j} Q_{jl} \hat L_j)  \otimes (\sum_k R_{lk} \hat R_k)

    The bond dimension is the size of the index $l$, which is truncated based on the rank of $\Gamma$.

    """
    assert non_red.shape == (len(term_row), len(term_col))
    assert k == 1
    non_red.data = factor[non_red.data - 1]
    gamma = non_red.todense()

    # gamma[:, p] = q @ r
    if gamma.shape[1] != 1:
        # normal qr
        q, r, p = scipy.linalg.qr(gamma, mode="economic", pivoting=True)
    else:
        # move the factor to q
        q = gamma
        r = np.array([1]).reshape(1, 1)
        p = np.array([0])
    # use relative tolerance for r since it's not normalized
    rtol = 1e-10
    rank = np.sum(np.abs(np.diag(r)) > np.abs(r[0][0]) * rtol)

    out_ops: List[List[OpTuple]] = [[] for _ in range(rank)]

    # use absolute tolerance for q since it's normalized
    atol = 1e-10
    for i, j in zip(*np.where(np.abs(q[:, :rank]) > atol)):
        symbol = term_row[i]
        qn = _compute_qn(in_ops_list, symbol, primary_ops, k)
        out_op = OpTuple(symbol, qn, factor=q[i, j])
        out_ops[j].append(out_op)

    # The R matrix truncated and sorted by pivoting
    r2 = r[:rank, np.argsort(p)]
    idx1, idx2 = np.where(np.abs(r2) > np.abs(r[0][0]) * rtol)
    new_factor = r2[(idx1, idx2)]
    # `new_table` can have more rows than `table`: this is a feature of the QR method
    # for example, the operator is ax + by, and QR might give a table with 4 rows
    # table  factor
    # a+b  x  1/2
    # a+b  y  1/2
    # a-b  x  -1/2
    # a-b  y  -1/2
    new_table = np.concatenate([idx1.reshape(-1, 1), [term_col[i] for i in idx2]], axis=1)
    return out_ops, new_table, new_factor


def _compute_qn(in_ops_list, symbol, primary_ops, k) -> int:
    qn = sum(in_ops[i][0].qn for in_ops, i in zip(in_ops_list, symbol[:-k]))
    qn += sum(primary_ops[i].qn for i in symbol[-k:])
    return qn


def _terms_to_table(model: Model, terms: List[Op], const: float):
    r"""
    constructing a general operator table
    according to model.model and model.order
    """

    table = []
    factor_list = []
    
    dummy_table_entry = []
    for b in model.basis:
        if b.multi_dof:
            dof = b.dof[0]
        else:
            dof = b.dof
        op = Op.identity(dof, qn_size=model.qn_size)
        dummy_table_entry.append(op)
    for op in terms:
        elem_ops, factor = op.split_elementary(model.dof_to_siteidx)
        table_entry = dummy_table_entry.copy()
        for elem_op in elem_ops:
            # it is ensured in `elem_op` every symbol is on the same site
            site_idx = model.dof_to_siteidx[elem_op.dofs[0]]
            table_entry[site_idx] = elem_op
        table.append(table_entry)
        factor_list.append(factor)

    # const
    if const != 0:
        table_entry = dummy_table_entry.copy()
        factor_list.append(const)
        table.append(table_entry)

    factor_list = np.array(factor_list)
    logger.debug(f"# of operator terms: {len(table)}")

    return table, factor_list


def _deduplicate_table(table, factor):

    # check the index of interaction could be represented with np.uint32
    assert table.shape[0] < np.iinfo(np.uint32).max

    # combine the same terms but with different factors(add them together)
    new_table, unique_inverse = np.unique(table, axis=0, return_inverse=True)
    # it is efficient to vectorize the operation that moves the rows and cols
    # and sum them together
    coord = np.array([[newidx, oldidx] for oldidx, newidx in enumerate(unique_inverse)])
    mask = scipy.sparse.csr_matrix((np.ones(len(coord)), (coord[:, 0], coord[:, 1])))
    factor = mask.dot(factor)

    # remove zeros
    mask = np.abs(factor) > (np.max(np.abs(factor)) * 1e-15)
    new_table = np.array(new_table)[mask]
    factor = factor[mask]

    return new_table, factor


def _transform_table(table, factor):
    """Transforms the table to integer table and combine duplicate terms."""

    # use np.uint32, np.uint16 to save memory
    max_uint16 = np.iinfo(np.uint16).max

    # translate the symbolic operator table to an easy to manipulate numpy array
    table = np.array(table)
    # unique operators with DoF names taken into consideration
    # The inclusion of DoF names is necessary for multi-dof basis.
    unique_op = OrderedDict.fromkeys(table.ravel())
    # Convert Set(table.ravel()) to List will change the Op order in list, OrderedDict made reproducible
    unique_op = list(unique_op.keys())

    # check the index of different operators could be represented with np.uint16
    assert len(unique_op) < max_uint16

    # Construct mapping from easy-to-manipulate integer to actual Op
    primary_ops = list(unique_op)

    op2idx = dict(zip(unique_op, range(len(unique_op))))
    new_table = np.vectorize(op2idx.get)(table).astype(np.uint16)

    del unique_op

    if __debug__:
        qn_size = len(table[0][0].qn)
        qn_table = np.array([[x.qn for x in ta] for ta in table])
        factor_table = np.array([[x.factor for x in ta] for ta in table])
        for idx in range(len(primary_ops)):
            coord = np.nonzero(new_table == idx)
            # check that op with the same symbol has the same factor and qn
            for j in range(qn_size):
                assert np.unique(qn_table[:, :, j][coord]).size == 1
            assert np.all(factor_table[coord] == factor_table[coord][0])

        del factor_table, qn_table

    table, factor = _deduplicate_table(new_table, factor)

    return table, factor, primary_ops


# translate the numbers into symbolic Matrix Operator
def compose_symbolic_mo(in_ops, out_ops, primary_ops):
    shape = [len(in_ops), len(out_ops)]
    mo = np.full(shape, None, dtype=object)
    for i, _ in np.ndenumerate(mo):
        mo[i] = []
    for iop, out_op in enumerate(out_ops):
        for composed_op in out_op:
            in_idx = composed_op.symbol[0]
            op = primary_ops[composed_op.symbol[1]]
            mo[in_idx][iop].append(composed_op.factor * op)
    return mo


# translate symbolic Matrix Operator to numerical matrix operator defined with certain basis
def symbolic_mo_to_numeric_mo(basis: BasisSet, mo, dtype):
    pdim = basis.nbas
    shape = list(mo.shape) + [pdim, pdim]
    mo_mat = np.zeros(shape, dtype=dtype)

    for i, terms in np.ndenumerate(mo):
        for term in terms:
            mo_mat[i] += basis.op_mat(term)

    axes = list(range(mo.ndim + 2))
    axes = axes[:-3] + axes[-2:] + [axes[-3]]
    return mo_mat.transpose(axes)


def _format_symbolic_mpo(symbolic_mpo):
    # debug tool. Used in the comment of Mpo.__init__

    # helper function
    def format_op(op: Op):
        op_str = op.symbol
        op_str = op_str.replace(r"^\dagger", "†")
        if op.factor != 1:
            op_str = f"{op.factor:.1e} * " + op_str
        return op_str

    result_str_list = []
    # print the MPO sites one by one
    for mo in symbolic_mpo:
        # firstly convert the site into an array of strings
        mo_str_array = np.full((len(mo), len(mo[0])), None)
        for irol, row in enumerate(mo):
            for icol, terms in enumerate(row):
                if len(terms) == 0:
                    terms_str = "0"
                else:
                    terms_str = " + ".join(format_op(op) for op in terms)
                mo_str_array[irol][icol] = terms_str
        # array of element length
        mo_str_length = np.vectorize(lambda x: len(x))(mo_str_array)
        max_length_per_col = mo_str_length.max(axis=0)
        # format each line
        lines = []
        for row in mo_str_array:
            terms_with_space = [term + " " * (max_length_per_col[icol] - len(term)) for icol, term in enumerate(row)]
            row_str = "   ".join(terms_with_space)
            lines.append("│ " + row_str + " │")
        # make it prettier
        if len(lines) != 1:
            lines[0] = "┏" + lines[0][1:-1] + "┓"
            lines[-1] = "┗" + lines[-1][1:-1] + "┛"
        # str of a single mo
        result_str_list.append("\n".join(lines))
    return "\n".join(result_str_list)


##############################################################################################
# symbolic MPO swapping algorithm


ExpandedOp = namedtuple("ExpandedOp", ["factor", "out_ops1_idx", "site1_op_idx", "site2_op_idx"])


def multiply_out_op_sum_list_by_out_op(l1: List, out_op: OpTuple):
    res = []
    for op_in_sum_list in l1:
        term = ExpandedOp(
            op_in_sum_list.factor * out_op.factor,
            op_in_sum_list.symbol[0], op_in_sum_list.symbol[1], out_op.symbol[1]
        )
        res.append(term)
    return res


def expand_out_op_sum_list(out_ops1: List, l2: List):
    res = []
    for out_op in l2:
        out_op_sum_list1 = out_ops1[out_op.symbol[0]]
        res.extend(multiply_out_op_sum_list_by_out_op(out_op_sum_list1, out_op))
    return res


def check_swap_consistency(new_out_ops2, new_out_ops3, out_ops3_expanded):
    # check consistency
    new_out_ops3_expanded: List[List[ExpandedOp]] = []
    for out_op_sum_list in new_out_ops3:
        new_out_ops3_expanded.append(expand_out_op_sum_list(new_out_ops2, out_op_sum_list))
    # item ordering: out_ops1, site1, site2, factor. (site indices are before swapping)
    # put the float-point factor to the last position for robust sorting
    swapped_new_out_ops3_expanded: List[List[Tuple]] = []
    for out_op_sum_list in new_out_ops3_expanded:
        grouped: Dict[Tuple, float] = defaultdict(int)
        for op in out_op_sum_list:
            grouped[(op.out_ops1_idx, op.site2_op_idx, op.site1_op_idx)] += op.factor
        swapped_new_out_ops3_expanded.append(_grouped_to_list(grouped))

    swapped_out_ops3_expanded: List[List[Tuple]] = []
    for out_op_sum_list in out_ops3_expanded:
        grouped: Dict[Tuple, float] = defaultdict(int)
        for op in out_op_sum_list:
            grouped[(op.out_ops1_idx, op.site1_op_idx, op.site2_op_idx)] += op.factor
        swapped_out_ops3_expanded.append(_grouped_to_list(grouped))

    # the following check ensures that the swapping logic is correct
    for row1, row2 in zip(swapped_out_ops3_expanded, swapped_new_out_ops3_expanded):
        if not len(row1) == len(row2):
            print("ok")
        assert len(row1) == len(row2)
        assert sorted(row1) == row1
        assert sorted(row2) == row2
        for op1, op2 in zip(sorted(row1), sorted(row2)):
            assert  op1[:-1] == op2[:-1]
            np.testing.assert_allclose(op1[-1], op2[-1], rtol=1e-8, atol=1e-11)


def _grouped_to_list(grouped: Dict[Tuple, float]) -> List[Tuple]:
    res: List[Tuple] = []
    max_v = max(np.abs(list(grouped.values())))
    for k, v in grouped.items():
        if abs(v) < abs(max_v) * 1e-10:
            continue
        res.append((k[0], k[2], k[1], v))
    res.sort()
    return res


def table_row_swapped_jw(row, primary_ops: List, op2idx: Dict):
    assert len(row) == 5
    assert row[-1] == 0
    # mapping rule
    # a1 -> a1 z2, a1^d -> a1^d z2
    # a2 -> z1 a2, a2^d -> z1 a2^d
    op1: Op = primary_ops[row[1]]
    op2: Op = primary_ops[row[2]]

    # remember: all possible operators: I Z + -
    # new sigma_z produced for dof1 by op2
    op1_new_sigma_z = (op1.split_symbol.count("sigma_+") + op1.split_symbol.count("sigma_-")) % 2
    # similar except by op2
    op2_new_sigma_z = (op2.split_symbol.count("sigma_+") + op2.split_symbol.count("sigma_-")) % 2
    # determine the coefficient
    op1_n_sigma_plus = op1.split_symbol.count("sigma_+")
    op1_n_sigma_minus = op1.split_symbol.count("sigma_-")
    assert op1_n_sigma_plus in [0, 1]
    assert op1_n_sigma_minus in [0, 1]
    n_permutes = op2_new_sigma_z * (op1_n_sigma_plus + op1_n_sigma_minus)
    coeff = (-1) ** n_permutes
    # cancel sigma_z as much as possible
    def prepend_sigma_z(op: Op):
        symbol_list = op.split_symbol
        if symbol_list[0] == "I":
            assert len(symbol_list) == 1
            new_op = Op("sigma_z", op.dofs[0], qn=0)
        elif symbol_list[0] == "sigma_z":
            if len(symbol_list) == 1:
                new_op = Op.identity(op.dofs[0])
            else:
                new_op = Op(" ".join(symbol_list[1:]), op.dofs[1:], qn=op.qn_list[1:])
        elif symbol_list[0] == "sigma_+" or symbol_list[0] == "sigma_-":
            new_op = Op("sigma_z " + op.symbol, [op.dofs[0]] + op.dofs, qn=[0] + op.qn_list)
        else:
            assert False
        return new_op
    if op1_new_sigma_z:
        new_op2 = prepend_sigma_z(op2)
    else:
        new_op2 = op2
    if op2_new_sigma_z:
        new_op1 = prepend_sigma_z(op1)
    else:
        new_op1 = op1

    if new_op1 not in op2idx:
        op2idx[new_op1] = len(primary_ops)
        primary_ops.append(new_op1)
    if new_op2 not in op2idx:
        op2idx[new_op2] = len(primary_ops)
        primary_ops.append(new_op2)

    return [row[0], op2idx[new_op1], op2idx[new_op2], row[3], row[4]], coeff


def table_and_factor_swapped_jw(table, factor, primary_ops: List):
    # modifies primary_ops in place !!
    new_table = []
    new_factor = []
    op2idx = {op: i for i, op in enumerate(primary_ops)}
    for row, factor_row in zip(table, factor):
        new_row, coeff = table_row_swapped_jw(row, primary_ops, op2idx)
        new_table.append(new_row)
        new_factor.append(coeff * factor_row)
    return np.array(new_table), np.array(new_factor)


def swap_site(out_ops_list, primary_ops: List, swap_jw: bool, algo="Hopcroft-Karp"):
    # the MPO at hand is - # - # -
    # the bond indices are 1, 2, 3 and the site indices are 1 and 2
    # the operators at each of the bond
    out_ops1, out_ops2, out_ops3 = out_ops_list

    # the expanded sum-of-product form of the operator represented by the two site MPO
    # i.e. the expanded form of out_ops3
    # the number of items is equal to the index of bond 3
    # each item [factor, out_ops[1]idx, primary_ops @ site 1, primary_ops @ site 2]
    out_ops3_expanded: List[List[ExpandedOp]] = []
    for out_op_sum_list in out_ops3:
        out_ops3_expanded.append(expand_out_op_sum_list(out_ops2, out_op_sum_list))
        # lots of del in the following code. The purposes are
        # 1. to indicate the variables are local
        # 2. to help avoid mis-using these variables in the following code (by typo etc.)
        del out_op_sum_list
    table = []
    factor = []
    # to be extended after primary_ops, used to label each bond for out_ops3
    auxiliary_dummy_primary_ops = []
    DummyOp = namedtuple("DummyOp", ["qn"])
    for out_ops3_sum_list in out_ops3:
        auxiliary_dummy_primary_ops.append(DummyOp(-out_ops3_sum_list[0].qn))
    n_primary_ops = len(primary_ops)

    if not swap_jw:
        # modify primary_ops in place can be avoided
        primary_ops = primary_ops.copy()
        primary_ops.extend(auxiliary_dummy_primary_ops)

    for i, out_ops3_sum_list in enumerate(out_ops3_expanded):
        for op in out_ops3_sum_list:
            # swap the sites and add two more columns at the end
            row = [op.out_ops1_idx, op.site2_op_idx, op.site1_op_idx, n_primary_ops + i, 0]
            table.append(row)
            factor.append(op.factor)
            del op, row
    table = np.array(table)
    factor = np.array(factor)
    table, factor = _deduplicate_table(table, factor)

    if swap_jw:
        # modifies primary_ops in place !!
        table, factor = table_and_factor_swapped_jw(table, factor, primary_ops)
        table[:, 3] = table[:, 3] + (len(primary_ops) - n_primary_ops)
        n_primary_ops = len(primary_ops)
        primary_ops = primary_ops.copy()
        primary_ops.extend(auxiliary_dummy_primary_ops)

    new_out_ops = _construct_symbolic_mpo(table, out_ops1, factor, primary_ops, algo=algo)
    assert len(new_out_ops) == 4
    new_out_ops1, new_out_ops2, new_out_ops3_unsorted = new_out_ops[:3]

    # sort the out operators
    new_out_ops3 = [None] * len(new_out_ops3_unsorted)
    assert len(new_out_ops3) == len(primary_ops) - n_primary_ops == len(auxiliary_dummy_primary_ops)
    assert len(new_out_ops[-1]) == 1
    for dummy_op in new_out_ops[-1][0]:
        idx1, idx2 = dummy_op.symbol
        idx2 -= n_primary_ops
        new_out_ops3[idx2] = new_out_ops3_unsorted[idx1]
        if dummy_op.factor != 1:
            for i, op in enumerate(new_out_ops3[idx2]):
                new_out_ops3[idx2][i] = OpTuple(symbol=op.symbol, qn=op.qn, factor=op.factor * dummy_op.factor)
        del dummy_op, idx1, idx2
    assert None not in new_out_ops3

    if not swap_jw:
        # if swap_jw == True, it's bound to fail
        check_swap_consistency(new_out_ops2, new_out_ops3, out_ops3_expanded)

    mo1 = compose_symbolic_mo(out_ops1, new_out_ops2, primary_ops)
    mo2 = compose_symbolic_mo(new_out_ops2, new_out_ops3, primary_ops)
    # print(_format_symbolic_mpo([mo1, mo2]))
    qn = [opsum[0].qn for opsum in new_out_ops2]
    return new_out_ops2, new_out_ops3, mo1, mo2, qn
