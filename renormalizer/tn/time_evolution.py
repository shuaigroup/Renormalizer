from math import factorial
from typing import Union, List
import logging

import scipy
from scipy import stats
import opt_einsum as oe

from renormalizer.mps.lib import compressed_sum
from renormalizer.mps.backend import np, xp
from renormalizer.mps.matrix import asxp
from renormalizer.lib import solve_ivp, expm_krylov
from renormalizer.utils.configs import EvolveMethod
from renormalizer.tn.node import TreeNodeTensor
from renormalizer.tn.tree import TTNO, TTNS, TTNEnviron, EVOLVE_METHODS
from renormalizer.tn.hop_expr import hop_expr1, hop_expr2


logger = logging.getLogger(__name__)


def time_derivative_vmf(ttns: TTNS, ttno: TTNO):
    # todo: benchmark and optimize
    environ_s = TTNEnviron(ttns, TTNO.identity(ttns.basis))
    environ_h = TTNEnviron(ttns, ttno)

    deriv_list = []
    for inode, node in enumerate(ttns.node_list):
        hop, _ = hop_expr1(node, ttns, ttno, environ_h)
        # idx1: children+physical, idx2: parent
        dim_parent = node.shape[-1]
        tensor = asxp(node.tensor)
        shape_2d = (-1, dim_parent)
        deriv = hop(tensor).reshape(shape_2d)
        if node.parent is not None:
            # apply projector and S^-1
            tensor = tensor.reshape(shape_2d)
            proj = tensor.conj() @ tensor.T
            ovlp = environ_s.node_list[inode].environ_parent.reshape(dim_parent, dim_parent)
            ovlp_inv = regularized_inversion(ovlp, ttns.evolve_config.reg_epsilon)
            deriv = oe.contract("bf, bg, fh -> gh", deriv, xp.eye(proj.shape[0]) - proj, asxp(ovlp_inv.T))
        qnmask = ttns.get_qnmask(node).reshape(deriv.shape)
        deriv_list.append(deriv[qnmask].ravel())
    return np.concatenate(deriv_list)


def regularized_inversion_debug(m, eps):
    # XXX: this is debug code
    evals, evecs = scipy.linalg.eigh(m)
    evals_i, evecs_i = scipy.linalg.eigh(m + np.eye(len(m)) * eps)
    scipy.linalg.inv(m + np.eye(len(m)) * eps)
    weight1 = 1
    evals = np.where(evals>0, evals, 0)
    weight2 = np.exp(-evals / eps)
    weight3 = np.random.rand(len(evals)) * 2
    evals1 = evals + eps * weight2
    # np.testing.assert_allclose(evals_i, evals1)
    evals2 = evals + eps * weight2
    evals3 = evals + eps * weight3
    new_evals = 1 / evals1
    # print(new_evals)
    return evecs @ np.diag(new_evals) @ evecs.T.conj()


def regularized_inversion(m, eps):
    m = m + np.eye(len(m)) * eps
    return scipy.linalg.pinv(m)


def regularized_inversion(m, eps):
    evals, evecs = scipy.linalg.eigh(m)
    weight = np.exp(-evals / eps)
    evals = evals + eps * weight
    return evecs @ np.diag(1 / evals) @ evecs.T.conj()


def evolve_tdvp_vmf(tts:TTNS, tto:TTNO, coeff:Union[complex, float], tau:float, first_step=None):

    def ivp_func(t, params):
        tts_t = TTNS.from_tensors(tts, params)
        return coeff * time_derivative_vmf(tts_t, tto)
    init_y = np.concatenate([node.tensor[tts.get_qnmask(node)].ravel() for node in tts.node_list])
    atol = tts.evolve_config.ivp_atol
    rtol = tts.evolve_config.ivp_rtol
    sol = solve_ivp(ivp_func, (0, tau), init_y, first_step=first_step, atol=atol, rtol=rtol)
    logger.info(f"VMF func called: {sol.nfev}. RKF steps: {len(sol.t)}")
    new_tts = TTNS.from_tensors(tts, sol.y[:, -1])
    new_tts.canonicalise()
    return new_tts


def evolve_prop_and_compress_tdrk4(ttns:TTNS, ttno:TTNO, coeff:Union[complex, float], tau:float):
    termlist = [ttns]
    for i in range(4):
        termlist.append(ttno.contract(termlist[-1]))
    for i, term in enumerate(termlist):
        term.scale((coeff * tau) ** i / factorial(i), inplace=True)
    return compressed_sum(termlist)


def evolve_tdvp_ps2(ttns:TTNS, ttno:TTNO, coeff:Union[complex, float], tau:float):
    # second order 2-site projector splitting
    tte = TTNEnviron(ttns, ttno)
    # in MPS language: left to right sweep
    local_steps1 = _tdvp_ps2_recursion_forward(ttns.root, ttns, ttno, tte, coeff, tau / 2)
    # in MPS language: right to left sweep
    local_steps2 = _tdvp_ps2_recursion_backward(ttns.root, ttns, ttno, tte, coeff, tau / 2)
    steps_stat = stats.describe(local_steps1 + local_steps2)
    logger.debug(f"TDVP-PS Krylov space: {steps_stat}")
    return ttns


def _tdvp_ps2_recursion_forward(snode: TreeNodeTensor,
                                ttns: TTNS,
                                ttno: TTNO,
                                ttne: TTNEnviron,
                                coeff:Union[complex, float],
                                tau:float) -> List[int]:
    """time evolution all of snode's children (without evolve snode!).
    The exception is when snode == tts.root, which is evolved.
    Cano center at snode when entering and leaving"""
    assert snode.children  # 2 site can't do only one node
    # todo: update to more general cases like truncation based on singular values
    m = ttns.compress_config.bond_dim_max_value
    local_steps = []
    for ichild, child in enumerate(snode.children):

        if child.children:
            # cano to child
            ttns.push_cano_to_child(snode, ichild)
            # update env
            ttne.update_1bond(child, ttns, ttno)
            # recursive time evolution
            local_steps_child = _tdvp_ps2_recursion_forward(child, ttns, ttno, ttne, coeff, tau)
            local_steps.extend(local_steps_child)

        # forward time evolution for snode + child
        ms2, j = evolve_2site(child, ttns, ttno, ttne, coeff, tau)
        local_steps.append(j)
        # cano to snode
        ttns.update_2site(child, ms2, m, cano_parent=True)
        # update env
        ttne.update_2site(child, ttns, ttno)
        # backward time evolution for snode
        if snode is ttns.root and ichild == len(snode.children) - 1:
            continue
        ms, j = evolve_1site(snode, ttns, ttno, ttne, coeff, -tau)
        snode.tensor = ms.reshape(snode.shape)
        local_steps.append(j)
        # update env
        ttne.update_1site(snode, ttns, ttno)
    return local_steps


def _tdvp_ps2_recursion_backward(snode: TreeNodeTensor,
                                 ttns: TTNS,
                                 ttno: TTNO,
                                 ttne: TTNEnviron,
                                 coeff:Union[complex, float],
                                 tau:float) -> List[int]:
    """time evolution all of snode's children (without evolve snode!).
    The exception is when snode == tts.root, which is evolved.
    Cano center at snode when entering and leaving"""
    assert snode.children  # 2 site can't do only one node
    # todo: update to more general cases like truncation based on singular values
    m = ttns.compress_config.bond_dim_max_value
    local_steps = []
    for ichild, child in reversed(list(enumerate(snode.children))):
        # backward time evolution for snode
        if not (snode is ttns.root and ichild == len(snode.children) - 1):
            ms, j = evolve_1site(snode, ttns, ttno, ttne, coeff, -tau)
            snode.tensor = ms.reshape(snode.shape)
            local_steps.append(j)
            # update env
            ttne.update_1site(snode, ttns, ttno)

        # forward time evolution for snode + child
        ms2, j = evolve_2site(child, ttns, ttno, ttne, coeff, tau)
        local_steps.append(j)
        # cano to snode
        ttns.update_2site(child, ms2, m, cano_parent=not child.children)
        # update env
        ttne.update_2site(child, ttns, ttno)

        if child.children:
            # recursive time evolution
            local_steps_child = _tdvp_ps2_recursion_backward(child, ttns, ttno, ttne, coeff, tau)
            local_steps.extend(local_steps_child)
            # cano to snode
            ttns.push_cano_to_parent(child)
            # update env
            ttne.update_1bond(child, ttns, ttno)
    return local_steps


def evolve_2site(snode: TreeNodeTensor, ttns: TTNS, ttno: TTNO, ttne: TTNEnviron, coeff:Union[complex, float], tau:float):
    # evolve snode and parent
    ms2 = ttns.merge_with_parent(snode)
    hop, _ = hop_expr2(snode, ttns, ttno, ttne)
    ms2_t, j = expm_krylov(
        lambda y: hop(y.reshape(ms2.shape)).ravel(),
        coeff * tau,
        ms2.ravel()
    )
    return ms2_t, j


def evolve_1site(snode: TreeNodeTensor, ttns: TTNS, ttno: TTNO, ttne: TTNEnviron, coeff:Union[complex, float], tau:float):
    ms = snode.tensor
    hop, _ = hop_expr1(snode, ttns, ttno, ttne)
    ms_t, j = expm_krylov(
        lambda y: hop(y.reshape(ms.shape)).ravel(),
        coeff * tau,
        ms.ravel()
    )
    return ms_t, j


EVOLVE_METHODS[EvolveMethod.tdvp_vmf] = evolve_tdvp_vmf
EVOLVE_METHODS[EvolveMethod.prop_and_compress_tdrk4] = evolve_prop_and_compress_tdrk4
EVOLVE_METHODS[EvolveMethod.tdvp_ps2] = evolve_tdvp_ps2
