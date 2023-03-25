import logging
from typing import List


import scipy

from renormalizer.lib import davidson
from renormalizer.mps.backend import primme, IMPORT_PRIMME_EXCEPTION, np
from renormalizer.mps.svd_qn import get_qn_mask
from renormalizer.tn.node import TreeNodeTensor
from renormalizer.tn.tree import TensorTreeState, TensorTreeOperator, TensorTreeEnviron
from renormalizer.tn.hop_expr import hop_expr2


logger = logging.getLogger(__name__)


def optimize_tts(tts: TensorTreeState, tto: TensorTreeOperator, procedure=None):
    if procedure is None:
        procedure = tts.optimize_config.procedure
    tte = TensorTreeEnviron(tts, tto)
    e_list = []
    for m, percent in procedure:
        # todo: better converge condition
        micro_e = optimize_recursion(tts.root, tts, tto, tte, m, percent)
        logger.info(f"Micro e: {micro_e}")
        e_list.append(micro_e[-1])
    return e_list


def optimize_recursion(snode: TreeNodeTensor, tts, tto, tte, m:int, percent:float=0) -> List[float]:
    assert snode.children  # 2 site can't do only one node
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

    cguess = tts.merge_with_parent(snode)
    qnmat = tts.get_qnmat(snode)[-1]
    qn_mask = get_qn_mask(qnmat, tts.qntot)
    cguess = cguess[qn_mask].ravel()
    expr, hdiag = hop_expr2(snode, tts, tto, tte)
    hdiag = hdiag[qn_mask].ravel()

    def hop(x):
        cstruct = vec2tensor(x, qn_mask)
        return expr(cstruct)[qn_mask].ravel()

    assert tts.optimize_config.nroots == 1
    algo:str = tts.optimize_config.algo
    e, c = eigh_iterative(hop, hdiag, cguess, algo)
    c = vec2tensor(c, qn_mask)
    return e, c


def eigh_iterative(hop, hdiag, cguess, algo):
    h_dim = len(hdiag)

    if algo == "davidson":
        precond = lambda x, e, *args: x / (hdiag - e + 1e-4)

        e, c = davidson(
            hop, cguess, precond, max_cycle=100, nroots=1, max_memory=64000, verbose=0
        )
    elif algo == "primme":
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
    elif algo == "arpack":
        A = scipy.sparse.linalg.LinearOperator((h_dim, h_dim), matvec=hop)
        e, c = scipy.sparse.linalg.eigsh(A, k=1, which="SA", v0=cguess)
        e = e[0]
    elif algo == "direct":
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
    else:
        assert False

    return e, c


def vec2tensor(c, qn_mask):
    cstruct = np.zeros(qn_mask.shape, dtype=c.dtype)
    np.place(cstruct, qn_mask, c)
    return cstruct
