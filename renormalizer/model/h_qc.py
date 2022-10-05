# -*- coding: utf-8 -*-

import itertools
import logging

import numpy as np

from renormalizer.model.op import Op
from renormalizer.model.basis import BasisHalfSpin

logger = logging.getLogger(__name__)

def read_fcidump(fname, norb):
    """
    from fcidump format electron integral to h_pq g_pqrs in arXiv:2006.02056 eq 18
    norb: number of spatial orbitals
    return sh spin-orbital 1-e integral
           aseri: 2-e integral after considering symmetry
           nuc: nuclear repulsion energy
    """
    eri = np.zeros((norb, norb, norb, norb))
    h = np.zeros((norb,norb))

    with open(fname, "r") as f:
        a = f.readlines()
        for line, info in enumerate(a):
            if line < 4:
                continue
            s  = info.split()
            integral, p, q, r, s = float(s[0]),int(s[1]),int(s[2]),int(s[3]),int(s[4])
            if r != 0:
                eri[p-1,q-1,r-1,s-1] = integral
                eri[q-1,p-1,r-1,s-1] = integral
                eri[p-1,q-1,s-1,r-1] = integral
                eri[q-1,p-1,s-1,r-1] = integral
            elif p != 0:
                h[p-1,q-1] = integral
                h[q-1,p-1] = integral
            else:
                nuc = integral

    sh, aseri = int_to_h(h, eri)

    logger.info(f"nuclear repulsion: {nuc}")

    return sh, aseri, nuc


def int_to_h(h, eri):
    nsorb = len(h) * 2
    seri = np.zeros((nsorb, nsorb, nsorb, nsorb))
    sh = np.zeros((nsorb, nsorb))
    for p, q, r, s in itertools.product(range(nsorb), repeat=4):
        # a_p^\dagger a_q^\dagger a_r a_s
        if p % 2 == s % 2 and q % 2 == r % 2:
            seri[p, q, r, s] = eri[p // 2, s // 2, q // 2, r // 2]

    for q, s in itertools.product(range(nsorb), repeat=2):
        if q % 2 == s % 2:
            sh[q, s] = h[q // 2, s // 2]

    aseri = np.zeros((nsorb, nsorb, nsorb, nsorb))
    for q, s in itertools.product(range(nsorb), repeat=2):
        for p, r in itertools.product(range(q), range(s)):
            # aseri[p,q,r,s] = seri[p,q,r,s] - seri[q,p,r,s]
            aseri[p, q, r, s] = seri[p, q, r, s] - seri[p, q, s, r]

    return sh, aseri

def qc_model(h1e, h2e, conserve_qn=True):
    """
    Ab initio electronic Hamiltonian in spin-orbitals
    h1e: sh above
    h2e: aseri above
    return model: "e_0", "e_1"... is according to the orbital index in sh and
    aseri
    """
    #------------------------------------------------------------------------
    # Jordan-Wigner transformation maps fermion problem into spin problem
    #
    # |0> => |alpha> and |1> => |beta >:
    #
    #    a_j   => Prod_{l=0}^{j-1}(sigma_z[l]) * sigma_+[j]
    #    a_j^+ => Prod_{l=0}^{j-1}(sigma_z[l]) * sigma_-[j]
    # j starts from 0 as in computer science convention to be consistent
    # with the following code.
    #------------------------------------------------------------------------

    norbs = h1e.shape[0]
    logger.info(f"spin norbs: {norbs}")
    assert np.all(np.array(h1e.shape) == norbs)
    assert np.all(np.array(h2e.shape) == norbs)

    # construct electronic creation/annihilation operators by Jordan-Wigner transformation
    a_ops = []
    a_dag_ops = []
    for j in range(norbs):
        # qn for each op will be processed in `process_op`
        sigma_z_list = [Op("Z", l) for l in range(j)]
        a_ops.append( Op.product(sigma_z_list + [Op("+", j)]) )
        a_dag_ops.append( Op.product(sigma_z_list + [Op("-", j)]) )

    ham_terms = []

    # helper function to process operators.
    # Remove "sigma_z sigma_z". Use {sigma_z, sigma_+} = 0
    # and {sigma_z, sigma_-} = 0 to simplify operators,
    # and set quantum number
    dof_to_siteidx = dict(zip(range(norbs), range(norbs)))
    if conserve_qn:
        qn_dict = {"+": -1, "-": 1, "Z": 0}
    else:
        qn_dict = {"+": 0, "-": 0, "Z": 0}
    def process_op(old_op: Op):
        old_ops, _ = old_op.split_elementary(dof_to_siteidx)
        new_ops = []
        for elem_op in old_ops:
            # move all sigma_z to the start of the operator
            # and cancel as many as possible
            n_sigma_z = elem_op.split_symbol.count("Z")
            n_non_sigma_z = 0
            n_permute = 0
            for simple_elem_op in elem_op.split_symbol:
                if simple_elem_op != "Z":
                    n_non_sigma_z += 1
                else:
                    n_permute += n_non_sigma_z
            # remove as many "sigma_z" as possible
            new_symbol = [s for s in elem_op.split_symbol if s != "Z"]
            if n_sigma_z % 2 == 1:
                new_symbol.insert(0, "Z")
            # this op is identity, discard it
            if not new_symbol:
                continue
            new_qn = [qn_dict[s] for s in new_symbol]
            new_dof_name = elem_op.dofs[0]
            new_ops.append(Op(" ".join(new_symbol), new_dof_name, (-1) ** n_permute, new_qn))
        return Op.product(new_ops)

    # 1-e terms
    for p, q in itertools.product(range(norbs), repeat=2):
        op = process_op(a_dag_ops[p] * a_ops[q])
        ham_terms.append(op * h1e[p, q])

    # 2-e terms.
    for q,s in itertools.product(range(norbs),repeat = 2):
        for p,r in itertools.product(range(q),range(s)):
            op = process_op(Op.product([a_dag_ops[p], a_dag_ops[q], a_ops[r], a_ops[s]]))
            ham_terms.append(op * h2e[p, q, r, s])

    if conserve_qn:
        sigmaqn = [0,1]
    else:
        sigmaqn=[0, 0]
    basis = [BasisHalfSpin(iorb, sigmaqn=sigmaqn) for iorb in range(norbs)]

    return basis, ham_terms
