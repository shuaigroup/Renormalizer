import logging
import itertools
from typing import Dict
from collections import defaultdict

import numpy as np
import scipy
import scipy.sparse

from renormalizer.model import MolList2, HolsteinModel
from renormalizer.mps.backend import xp
from renormalizer.mps.matrix import moveaxis, tensordot, asnumpy
from renormalizer.mps.mp import MatrixProduct
from renormalizer.mps import svd_qn
from renormalizer.mps.lib import update_cv
from renormalizer.lib import bipartite_vertex_cover
from renormalizer.utils import Quantity, Op
from renormalizer.utils.elementop import (
    construct_ph_op_dict,
    construct_e_op_dict,
    ph_op_matrix,
)


logger = logging.getLogger(__name__)


def symbolic_mpo(table, factor, algo="Hopcroft-Karp"):
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
    logger.debug(f"symbolic mpo algorithm: {algo}")
    # use np.uint32, np.uint16 to save memory
    max_uint32 = np.iinfo(np.uint32).max
    max_uint16 = np.iinfo(np.uint16).max
    
    nsite = len(table[0])
    logger.debug(f"Input operator terms: {len(table)}")
    # Simplest case. Cut to the chase
    if len(table) == 1:
        # The first layer: number of sites. The 2nd and 3rd layer: in and out virtual bond
        # the 4th layer: operator sums
        mpo: List[List[List[List[[Op]]]]] = []
        mpoqn = [[0]]
        for op in table[0]:
            mpo.append([[[op]]])
            qn = mpoqn[-1][0] + op.qn
            mpoqn.append([qn])
        old_op = mpo[-1][0][0][0]
        mpo[-1][0][0][0] = Op(old_op.symbol, old_op.qn, old_op.factor * factor[0])
        qntot = qn
        mpoqn[-1] = [0]
        qnidx = len(mpo) - 1
        return mpo, mpoqn, qntot, qnidx
    # translate the symbolic operator table to an easy to manipulate numpy array
    # extract the op symbol, qn, factor to a numpy array
    symbol_table = np.array([[x.symbol for x in ta] for ta in table])
    qn_table = np.array([[x.qn for x in ta] for ta in table])
    factor_table = np.array([[x.factor for x in ta] for ta in table])
    
    new_table = np.zeros((len(table), nsite),dtype=np.uint16)
    
    primary_ops = {}
    # check that op with the same symbol has the same factor and qn
    unique_symbol = np.unique(symbol_table)
    # check the index of different operators could be represented with np.uint16
    assert len(unique_symbol) < max_uint16
    for idx, s in enumerate(unique_symbol):
        coord = np.nonzero(symbol_table == s)
        assert np.unique(qn_table[coord]).size == 1
        assert np.all(factor_table[coord] == factor_table[coord][0])
        new_table[coord] = idx
        primary_ops[idx] = table[coord[0][0]][coord[1][0]]   
    
    del symbol_table, factor_table, qn_table
    
    # combine the same terms but with different factors(add them together)
    unique_term, unique_inverse = np.unique(new_table, axis=0, return_inverse=True)
    # it is efficient to vectorize the operation that moves the rows and cols
    # and sum them together
    coord = np.array([[newidx, oldidx] for oldidx, newidx in enumerate(unique_inverse)])
    mask = scipy.sparse.csr_matrix((np.ones(len(coord)), (coord[:,0], coord[:,1])))
    factor = mask.dot(factor)
    
    # add the first and last column for convenience
    ta = np.zeros((unique_term.shape[0],1),dtype=np.uint16)
    table = np.concatenate((ta, unique_term, ta), axis=1)
    logger.debug(f"After combination of the same terms: {table.shape[0]}")
    # check the index of interaction could be represented with np.uint32
    assert table.shape[0] < max_uint32
    
    del unique_term, unique_inverse

    mpo = []
    mpoqn = [[0]]

    in_ops = [[Op.identity()]]

    for isite in range(nsite):
        # split table into the row and col part
        term_row, row_unique_inverse = np.unique(table[:,:2], axis=0, return_inverse=True)
        term_col, col_unique_inverse = np.unique(table[:,2:], axis=0, return_inverse=True)
        
        # get the non_redudant ops
        # the +1 trick is to use the csr sparse matrix format
        non_red = scipy.sparse.diags(np.arange(1,table.shape[0]+1), format="csr", dtype=np.uint32)
        coord = np.array([[newidx, oldidx] for oldidx, newidx in enumerate(row_unique_inverse)])
        mask = scipy.sparse.csr_matrix((np.ones(len(coord), dtype=np.uint32), (coord[:,0], coord[:,1])))
        non_red = mask.dot(non_red)
        coord = np.array([[oldidx, newidx] for oldidx, newidx in enumerate(col_unique_inverse)])
        mask = scipy.sparse.csr_matrix((np.ones(len(coord), dtype=np.uint32), (coord[:,0], coord[:,1])))
        non_red = non_red.dot(mask)
        # use sparse matrix to represent non_red will be inefficient a little
        # bit compared to dense matrix, but saves a lot of memory when the
        # number of terms is huge
        # logger.info(f"isite: {isite}, bipartite graph size: {non_red.shape}")

        # select the reserved ops
        out_ops = []
        new_table = []
        new_factor = []
        
        bigraph = []
        if non_red.shape[0] < non_red.shape[1]:
            for i in range(non_red.shape[0]):
                bigraph.append(non_red.indices[non_red.indptr[i]:non_red.indptr[i+1]])
            rowbool, colbool = bipartite_vertex_cover(bigraph, algo=algo)
        else:
            non_red_csc = non_red.tocsc()
            for i in range(non_red.shape[1]):
                bigraph.append(non_red_csc.indices[non_red_csc.indptr[i]:non_red_csc.indptr[i+1]])
            colbool, rowbool = bipartite_vertex_cover(bigraph, algo=algo)

        row_select = np.nonzero(rowbool)[0]
        col_select = np.nonzero(colbool)[0]
        if len(row_select) > 0:
            assert np.amax(row_select) < non_red.shape[0]
        if len(col_select) > 0:
            assert np.amax(col_select) < non_red.shape[1]
        
        for row_idx in row_select:
            # construct out_op
            # dealing with row (left side of the table). One row corresponds to multiple cols.
            # Produce one out operator and multiple new_table entries
            symbol = term_row[row_idx]
            qn = in_ops[term_row[row_idx][0]][0].qn + primary_ops[term_row[row_idx][1]].qn
            out_op = Op(symbol, qn, factor=1.0)
            out_ops.append([out_op])
            
            col_link = non_red.indices[non_red.indptr[row_idx]:non_red.indptr[row_idx+1]]
            stack = np.array([len(out_ops)-1]*len(col_link), dtype=np.uint16).reshape(-1,1)
            new_table.append(np.hstack((stack,term_col[col_link])))
            new_factor.append(factor[non_red[row_idx, col_link].toarray()-1])
        
            non_red.data[non_red.indptr[row_idx]:non_red.indptr[row_idx+1]] = 0
        non_red.eliminate_zeros()
        
        nonzero_row_idx, nonzero_col_idx = non_red.nonzero()
        for col_idx in col_select:
            
            out_ops.append([])
            # complementary operator
            # dealing with column (right side of the table). One col correspond to multiple rows.
            # Produce multiple out operators and one new_table entry
            non_red_one_col = non_red[:, col_idx].toarray()
            for i in nonzero_row_idx[np.nonzero(nonzero_col_idx == col_idx)[0]]:
                symbol = term_row[i]
                qn = in_ops[term_row[i][0]][0].qn + primary_ops[term_row[i][1]].qn
                out_op = Op(symbol, qn, factor=factor[non_red_one_col[i]-1])
                out_ops[-1].append(out_op)
            
            new_table.append(np.array([len(out_ops)-1] + list(term_col[col_idx]), dtype=np.uint16).reshape(1,-1))
            new_factor.append(1.0)
            
            # it is not necessary to remove the column nonzero elements
            #non_red[:, col_idx] = 0
            #non_red.eliminate_zeros()
            
        # translate the numpy array back to symbolic mpo
        mo = [[[] for o in range(len(out_ops))] for i in range(len(in_ops))]
        moqn = []

        for iop, out_op in enumerate(out_ops):
            for composed_op in out_op:
                in_idx = composed_op.symbol[0]
                symbol = primary_ops[composed_op.symbol[1]].symbol
                qn = primary_ops[composed_op.symbol[1]].qn
                if isite != nsite-1:
                    factor = composed_op.factor
                else:
                    factor = composed_op.factor*new_factor[0]
                new_op = Op(symbol, qn, factor)
                mo[in_idx][iop].append(new_op)
            moqn.append(out_op[0].qn)

        mpo.append(mo)
        mpoqn.append(moqn)
        # reconstruct the table in new operator 
        table = np.concatenate(new_table)
        # check the number of incoming operators could be represent as np.uint16
        assert len(out_ops) <= max_uint16
        factor = np.concatenate(new_factor, axis=None)
        #debug
        #logger.debug(f"in_ops: {in_ops}")
        #logger.debug(f"out_ops: {out_ops}")
        #logger.debug(f"new_factor: {new_factor}")
        
        in_ops = out_ops
   
    qntot = mpoqn[-1][0] 
    mpoqn[-1] = [0]
    qnidx = len(mpo)-1
    
    return mpo, mpoqn, qntot, qnidx


def add_idx(symbol, idx):
    symbols = symbol.split(" ")
    for i in range(len(symbols)):
        symbols[i] = symbols[i]+f"_{idx}"
    return " ".join(symbols)


def _model_translator_general_model(mol_list, model, const=Quantity(0.)):
    r"""
    constructing a general operator table
    according to mol_list.model and mol_list.order
    """
    
    factor = []
    table = []
    nsite = mol_list.nsite
    order = mol_list.order
    basis = mol_list.basis
    
    # clean up 
    # for example 
    # in simple_e case
    # ("e_0", "e_0"):[(Op("a^\dagger"), Op("a"), factor)] is not a standard
    # format for general model since "e_0" is a single site, the standard is
    # ("e_0"):[(Op("a^\dagger a"), factor)]
    # in multi_e case
    # ("e_0", "e_1"):[(Op("a^\dagger"), Op("a"), factor)] is not a standard
    # format for general model since "e_0" "e_1" is a single site, the standard
    # is ("e_0",):[(Op("a^\dagger_0 a_1"), factor)] 
    # or 
    # ("e_0",):[(Op("a^\dagger"), factor)] is not a standard
    # format for general model since "e_0" is a multi_e site, the standard
    # is ("e_0",):[(Op("a^\dagger_0"), factor)] 
    
    model_new = defaultdict(list)
    for model_key, model_value in model.items():

        # site index of the dofs for each term in the model
        dof_site_idx = [order[k] for k in model_key]

        dof_site_idx_unique = len(dof_site_idx) == len(set(dof_site_idx))
        if not dof_site_idx_unique or mol_list.multi_dof_basis:
            # keys: site index; values: index of the dofs (in terms of the model key) on the site index
            dofdict = defaultdict(list)
            for idof, dof in enumerate(model_key):
                dofdict[order[dof]].append(idof)

            new_key = tuple(dofdict.keys())
            new_value = []
            for op_term in model_value:
                # op_term is of format (Op, Op,..., float)
                new_term = []
                for iops in dofdict.values():
                    symbols = []
                    qn = 0
                    for iop in iops:
                        dof_name, dof_op = model_key[iop], op_term[iop]
                        dof_basis = basis[order[dof_name]]
                        if dof_basis.multi_dof and "_" not in dof_op.symbol:
                            # add the index to the operator in multi electron case
                            # for example, "a^\dagger a" on a single e_dof
                            # "a^\dagger" "a" on two different e_dofs
                            symbols.append(add_idx(dof_op.symbol, dof_name.split("_")[1]))
                        else:
                            symbols.append(dof_op.symbol)
                        qn += dof_op.qn
                    op = Op(" ".join(symbols), qn)
                    new_term.append(op)

                if isinstance(op_term[-1], Quantity):
                    raise TypeError("term factor should be float instead of Quantity")
                new_term.append(op_term[-1])
                new_value.append(tuple(new_term))

            model_new[new_key] += new_value
        else:
            model_new[tuple(dof_site_idx)] += model_value

    model = model_new

    for dof, model_value in model.items():
        op_factor = np.array([term[-1] for term in model_value])
        nonzero_idx = np.nonzero(op_factor != 0)[0]
        mapping = {j:i for i,j in enumerate(dof)}
        for idx in nonzero_idx:
            op_term = model_value[idx]
            # note that all the list element point to the same identity
            # it will not destroy the following operation since identity is not changed
            identity = Op.identity()
            ta = [op_term[mapping[isite]] if isite in mapping.keys() else identity for isite in range(nsite)]
            table.append(ta)
            factor.append(op_term[-1])
        
    # const
    if const != 0:
        ta = [Op.identity() for i in range(nsite)]
        factor.append(const.as_au())
        table.append(ta)
    
    factor = np.array(factor)
    logger.debug(f"# of operator terms: {len(table)}")

    return table, factor


class Mpo(MatrixProduct):
    """
    Matrix product operator (MPO)
    """

    @classmethod
    def exact_propagator(cls, mol_list: HolsteinModel, x, space="GS", shift=0.0):
        """
        construct the GS space propagator e^{xH} exact MPO
        H=\\sum_{in} \\omega_{in} b^\\dagger_{in} b_{in}
        fortunately, the H is local. so e^{xH} = e^{xh1}e^{xh2}...e^{xhn}
        the bond dimension is 1
        shift is the a constant for H+shift
        """
        assert space in ["GS", "EX"]

        mpo = cls()
        if np.iscomplex(x):
            mpo.to_complex(inplace=True)
        mpo.mol_list = mol_list

        for imol, mol in enumerate(mol_list):
            if mol_list.scheme < 4:
                mo = np.eye(2).reshape(1, 2, 2, 1)
                mpo.append(mo)
            elif mol_list.scheme == 4:
                if len(mpo) == mol_list.order["e_0"]:
                    n = mol_list.mol_num
                    mpo.append(np.eye(n+1).reshape(1, n+1, n+1, 1))
            else:
                assert False

            for ph in mol.ph_list:

                if space == "EX":
                    # for the EX space, with quasiboson algorithm, the b^\dagger + b
                    # operator is not local anymore.
                    ph_pbond = ph.pbond
                    # construct the matrix exponential by diagonalize the matrix first
                    phop = construct_ph_op_dict(ph_pbond)

                    h_mo = (
                        phop[r"b^\dagger b"] * ph.omega[0]
                        + phop[r"(b^\dagger + b)^3"] * ph.term30
                        + phop[r"b^\dagger + b"] * (ph.term10 + ph.term11)
                        + phop[r"(b^\dagger + b)^2"] * (ph.term20 + ph.term21)
                        + phop[r"(b^\dagger + b)^3"] * (ph.term31 - ph.term30)
                    )

                    w, v = scipy.linalg.eigh(h_mo)
                    h_mo = np.diag(np.exp(x * w))
                    h_mo = v.dot(h_mo)
                    h_mo = h_mo.dot(v.T)
                    mo = h_mo.reshape(1, ph_pbond, ph_pbond, 1)

                    mpo.append(mo)

                elif space == "GS":
                    anharmo = False
                    # for the ground state space, yet doesn't support 3rd force
                    # potential quasiboson algorithm
                    ph_pbond = ph.pbond
                    for i in range(len(ph.force3rd)):
                        anharmo = not np.allclose(
                            ph.force3rd[i] * ph.dis[i] / ph.omega[i], 0.0
                        )
                        if anharmo:
                            break
                    if not anharmo:
                        d = np.exp(
                                x
                                * ph.omega[0]
                                * np.arange(ph_pbond)
                            )
                        mo = np.diag(d).reshape(1, ph_pbond, ph_pbond, 1)
                        mpo.append(mo)
                    else:
                        # construct the matrix exponential by diagonalize the matrix first
                        phop = construct_ph_op_dict(ph_pbond)
                        h_mo = (
                            phop[r"b^\dagger b"] * ph.omega[0]
                            + phop[r"(b^\dagger + b)^3"] * ph.term30
                        )
                        w, v = scipy.linalg.eigh(h_mo)
                        h_mo = np.diag(np.exp(x * w))
                        h_mo = v.dot(h_mo)
                        h_mo = h_mo.dot(v.T)

                        mo = np.zeros([1, ph_pbond, ph_pbond, 1])
                        mo[0, :, :, 0] = h_mo

                        mpo.append(mo)
                else:
                    assert False
        # shift the H by plus a constant

        mpo.qn = [[0]] * (len(mpo) + 1)
        mpo.qnidx = len(mpo) - 1
        mpo.qntot = 0

        # np.exp(shift * x) is usually very large
        mpo = mpo.scale(np.exp(shift * x), inplace=True)

        return mpo

    @classmethod
    def onsite(cls, mol_list: MolList2, opera, dipole=False, mol_idx_set=None):

        qn_dict= {"a":-1, r"a^\dagger":1, r"a^\dagger a":0, "sigma_x":0, "sigma_z": 0}
        if mol_idx_set is None:
            mol_idx_set = range(len(mol_list.e_dofs))
        model = {}
        for idx in mol_idx_set:
            if dipole:
                factor = mol_list.dipole[(f"e_{idx}",)]
            else:
                factor = 1.
            model[(f"e_{idx}",)] = [(Op(opera, qn_dict[opera]),factor)]

        mpo = cls.general_mpo(mol_list, model=model)

        return mpo

    @classmethod
    def ph_onsite(cls, mol_list: HolsteinModel, opera: str, mol_idx:int, ph_idx=0):
        assert opera in ["b", r"b^\dagger", r"b^\dagger b"]
        assert mol_list.map is not None

        model = {(mol_list.map[(mol_idx, ph_idx)],): [(Op(opera,0), 1.0)]}
        mpo = cls.general_mpo(mol_list, model=model)

        return mpo

    @classmethod
    def intersite(cls, mol_list: HolsteinModel, e_opera: dict, ph_opera: dict, scale:
            Quantity=Quantity(1.)):
        """ construct the inter site MPO
        
        Parameters
        ----------
        mol_list : HolsteinModel
            the molecular information
        e_opera:
            the electronic operators. {imol: operator}, such as {1:"a", 3:r"a^\dagger"}
        ph_opera:
            the vibrational operators. {(imol, iph): operator}, such as {(0,5):"b"}
        scale: Quantity
            scalar to scale the mpo

        Note
        -----
        the operator index starts from 0,1,2...
        
        """
            
        assert mol_list.map is not None
        qn_dict= {"a":-1, r"a^\dagger":1, r"a^\dagger a":0, "sigma_x":0}

        key = []
        ops = []
        for e_key, e_op in e_opera.items():
            key.append(f"e_{e_key}")
            ops.append(Op(e_op, qn_dict[e_op]))
        for v_key, v_op in ph_opera.items():
            key.append(mol_list.map[v_key])
            ops.append(Op(v_op, 0))
        ops.append(scale.as_au())

        model = {tuple(key):[tuple(ops)]}
        mpo = cls(mol_list, model=model)

        return mpo


    @classmethod
    def finiteT_cv(cls, mol_list, nexciton, m_max, spectratype,
                   percent=1.0):
        np.random.seed(0)

        X = cls()
        X.mol_list = mol_list
        if spectratype == "abs":
            # quantum number index, |1><0|
            tag_1, tag_2 = 0, 1
        elif spectratype == "emi":
            # quantum number index, |0><1|
            tag_1, tag_2 = 1, 0
        X.qn = [[[0, 0]]]
        for ix in range(mol_list.nsite - 1):
            X.qn.append(None)
        X.qn.append([[0, 0]])
        dim_list = [1]

        for ix in range(mol_list.nsite - 1):
            if mol_list.is_electron(ix):
                qnbig = list(itertools.chain.from_iterable(
                    [np.add(y, [0, 0]), np.add(y, [0, 1]),
                     np.add(y, [1, 0]), np.add(y, [1, 1])]
                    for y in X.qn[ix]))
            else:
                qnbig = list(itertools.chain.from_iterable(
                    [x] * (mol_list.pbond_list[ix] ** 2) for x in X.qn[ix]))
            # print('qnbig', qnbig)
            u_set = []
            s_set = []
            qnset = []
            if spectratype != "conductivity":
                fq = list(itertools.chain.from_iterable([y[tag_1]] for y in qnbig))
                for iblock in range(min(fq), nexciton+1):
                    indices = [i for i, y in enumerate(qnbig) if
                               ((y[tag_1] == iblock) and (y[tag_2] == 0))]
                    if len(indices) != 0:
                        np.random.seed(0)
                        a: np.ndarray = np.random.random([len(indices), len(indices)]) - 0.5
                        a = a + a.T
                        s, u = scipy.linalg.eigh(a=a)
                        u_set.append(svd_qn.blockrecover(indices, u, len(qnbig)))
                        s_set.append(s)
                        if spectratype == "abs":
                            qnset += [iblock, 0] * len(indices)
                        elif spectratype == "emi":
                            qnset += [0, iblock] * len(indices)
            else:
                fq1 = list(itertools.chain.from_iterable([y[0]] for y in qnbig))
                fq2 = list(itertools.chain.from_iterable([y[1]] for y in qnbig))
                # print('fq1, fq2', fq1, fq2)
                for iblock in range(min(fq1), nexciton+1):
                    for jblock in range(min(fq2), nexciton+1):
                        # print('iblock', iblock, jblock)
                        indices = [i for i, y in enumerate(qnbig) if
                                   ((y[0] == iblock) and (y[1] == jblock))]
                        # print('indices', indices)
                        if len(indices) != 0:
                            a: np.ndarray = np.random.random([len(indices), len(indices)]) - 0.5
                            a = a + a.T
                            s, u = scipy.linalg.eigh(a=a)
                            u_set.append(svd_qn.blockrecover(indices, u, len(qnbig)))
                            s_set.append(s)
                            qnset += [iblock, jblock] * len(indices)
                            # print('iblock', iblock)
            list_qnset = []
            for i in range(0, len(qnset), 2):
                list_qnset.append([qnset[i], qnset[i + 1]])
            qnset = list_qnset
            # print('qnset', qnset)
            u_set = np.concatenate(u_set, axis=1)
            s_set = np.concatenate(s_set)
            # print('uset', u_set.shape)
            # print('s_set', s_set.shape)
            x, xdim, xqn, compx = update_cv(u_set, s_set, qnset, None, nexciton, m_max, spectratype, percent=percent)
            dim_list.append(xdim)
            X.qn[ix + 1] = xqn
            x = x.reshape(dim_list[-2], mol_list.pbond_list[ix], mol_list.pbond_list[ix], dim_list[ix + 1])
            X.append(x)
        dim_list.append(1)
        X.append(np.random.random([dim_list[-2], mol_list.pbond_list[-1],
                                  mol_list.pbond_list[-1], dim_list[-1]]))
        X.qnidx = len(X) - 1
        X.to_right = False
        X.qntot = nexciton
        # print('dim', [X[i].shape for i in range(len(X))])
        return X


    @classmethod
    def identity(cls, mol_list: MolList2):
        mpo = cls()
        mpo.mol_list = mol_list
        for p in mol_list.pbond_list:
            mpo.append(np.eye(p).reshape(1, p, p, 1))
        mpo.build_empty_qn()
        return mpo
    
    @classmethod
    def general_mpo(cls, mol_list, offset=Quantity(0.), model=None):
        """
        MolList2 or MolList with MolList2 parameters
        """
        return cls(mol_list, offset, model)
    
    def _general_mpo(self, mol_list, const=Quantity(0.), model=None):
        # construct a real mpo matrix elements into mpo shell

        assert len(self) == 0

        if model is None:
            model = mol_list.model

        table, factor = _model_translator_general_model(mol_list, model, const)

        self.dtype = factor.dtype

        mpo_symbol, mpo_qn, qntot, qnidx = symbolic_mpo(table, factor)
        # todo: elegant way to express the symbolic mpo
        #logger.debug(f"symbolic mpo: \n {np.array(mpo_symbol)}")
        self.mol_list = mol_list
        self.qnidx = qnidx
        self.qntot = qntot
        self.qn = mpo_qn
        self.to_right = False

        # evaluate the symbolic mpo
        assert mol_list.basis is not None
        basis = mol_list.basis

        for impo, mo in enumerate(mpo_symbol):
            pdim = basis[impo].nbas
            nrow, ncol = len(mo), len(mo[0])
            mo_mat = np.zeros((nrow, pdim, pdim, ncol), dtype=self.dtype)

            for irow, icol in itertools.product(range(nrow), range(ncol)):
                for term in mo[irow][icol]:
                    mo_mat[irow,:,:,icol] += basis[impo].op_mat(term)

            self.append(mo_mat)


    def __init__(self, mol_list: MolList2 = None, offset: Quantity = Quantity(0), model: Dict = None):

        """
        todo: document
        """
        # leave the possibility to construct MPO by hand
        super(Mpo, self).__init__()
        if mol_list is None:
            return
        if not isinstance(offset, Quantity):
            raise ValueError("offset must be Quantity object")

        self.offset = offset.as_au()
        self._general_mpo(mol_list, const=-offset, model=model)


    def _get_sigmaqn(self, idx):
        array_up = self.mol_list.basis[idx].sigmaqn
        return np.subtract.outer(array_up, array_up)

    @property
    def is_mps(self):
        return False

    @property
    def is_mpo(self):
        return True

    @property
    def is_mpdm(self):
        return False

    def metacopy(self):
        new = super().metacopy()
        # some mpo may not have these things
        attrs = ["scheme", "offset"]
        for attr in attrs:
            if hasattr(self, attr):
                setattr(new, attr, getattr(self, attr))
        return new

    @property
    def dummy_qn(self):
        return [[0] * dim for dim in self.bond_dims]

    @property
    def digest(self):
        return np.array([mt.var() for mt in self]).var()

    def promote_mt_type(self, mp):
        if self.is_complex and not mp.is_complex:
            mp.to_complex(inplace=True)
        return mp

    def apply(self, mp: MatrixProduct, canonicalise: bool=False) -> MatrixProduct:
        # todo: use meta copy to save time, could be subtle when complex type is involved
        # todo: inplace version (saved memory and can be used in `hybrid_exact_propagator`)
        # the mol_list is the same as the mps.mol_list
        new_mps = self.promote_mt_type(mp.copy())
        if mp.is_mps:
            # mpo x mps
            for i, (mt_self, mt_other) in enumerate(zip(self, mp)):
                assert mt_self.shape[2] == mt_other.shape[1]
                # mt=np.einsum("apqb,cqd->acpbd",mpo[i],mps[i])
                mt = xp.moveaxis(
                    tensordot(mt_self.array, mt_other.array, axes=([2], [1])), 3, 1
                )
                mt = mt.reshape(
                    (
                        mt_self.shape[0] * mt_other.shape[0],
                        mt_self.shape[1],
                        mt_self.shape[-1] * mt_other.shape[-1],
                    )
                )
                new_mps[i] = mt
        elif mp.is_mpo or mp.is_mpdm:
            # mpo x mpo
            for i, (mt_self, mt_other) in enumerate(zip(self, mp)):
                assert mt_self.shape[2] == mt_other.shape[1]
                # mt=np.einsum("apqb,cqrd->acprbd",mt_s,mt_o)
                mt = xp.moveaxis(
                    tensordot(mt_self.array, mt_other.array, axes=([2], [1])),
                    [-3, -2],
                    [1, 3],
                )
                mt = mt.reshape(
                    (
                        mt_self.shape[0] * mt_other.shape[0],
                        mt_self.shape[1],
                        mt_other.shape[2],
                        mt_self.shape[-1] * mt_other.shape[-1],
                    )
                )
                new_mps[i] = mt
        else:
            assert False
        orig_idx = new_mps.qnidx
        new_mps.move_qnidx(self.qnidx)
        qn = self.qn if not mp.use_dummy_qn else self.dummy_qn
        new_mps.qn = [
            np.add.outer(np.array(qn_o), np.array(qn_m)).ravel().tolist()
            for qn_o, qn_m in zip(qn, new_mps.qn)
        ]
        new_mps.qntot += self.qntot
        new_mps.move_qnidx(orig_idx)
        # concerns about whether to canonicalise:
        # * canonicalise helps to keep mps in a truly canonicalised state
        # * canonicalise comes with a cost. Unnecessary canonicalise (for example in P&C evolution and
        #   expectation calculation) hampers performance.
        if canonicalise:
            new_mps.canonicalise()
        return new_mps

    def contract(self, mps, algo="svd"):
        r""" an approximation of mpo @ mps/mpdm/mpo
        
        Parameters
        ----------
        mps : `Mps`, `Mpo`, `MpDm`
        algo: str, optional
            The algorithm to compress mpo @ mps/mpdm/mpo.  It could be ``svd``
            (default) and ``variational``. 
        
        Returns
        -------
        new_mps : `Mps`
            an approximation of mpo @ mps/mpdm/mpo. The input ``mps`` is not
            overwritten.

        See Also
        --------
        renormalizer.mps.mp.MatrixProduct.compress : svd compression.
        renormalizer.mps.mp.MatrixProduct.variational_compress : variational
            compression.


        """
        if algo == "svd":
            # mapply->canonicalise->compress
            new_mps = self.apply(mps)
            new_mps.canonicalise()
            new_mps.compress()
        elif algo == "variational":
            new_mps = mps.variational_compress(self)
        else:
            assert False

        return new_mps

    def conj_trans(self):
        new_mpo = self.metacopy()
        for i in range(new_mpo.site_num):
            new_mpo[i] = moveaxis(self[i], (1, 2), (2, 1)).conj()
        new_mpo.qn = [[-i for i in mt_qn] for mt_qn in new_mpo.qn]
        return new_mpo

    def full_operator(self):
        dim = np.prod(self.pbond_list)
        if 20000 < dim:
            raise ValueError("operator too large")
        res = np.ones((1, 1, 1, 1))
        for mt in self:
            dim1 = res.shape[1] * mt.shape[1]
            dim2 = res.shape[2] * mt.shape[2]
            dim3 = mt.shape[-1]
            res = np.tensordot(res, mt.array, axes=1).transpose((0, 1, 3, 2, 4, 5)).reshape(1, dim1, dim2, dim3)
        return res[0, :, :, 0]

    def is_hermitian(self):
        full = self.full_operator()
        return np.allclose(full.conj().T, full, atol=1e-7)

    def __matmul__(self, other):
        return self.apply(other)

    @classmethod
    def from_mp(cls, mol_list, mp):
        # mpo from matrix product
        mpo = cls()
        mpo.mol_list = mol_list
        for mt in mp:
            mpo.append(mt)
        mpo.build_empty_qn()
        return mpo
    
    @property
    def dmrg_norm(self) -> float:
        res = np.sqrt(self.conj().dot(self).real)
        return float(res.real)
    
