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



EVOLVE_METHODS[EvolveMethod.tdvp_vmf] = evolve_tdvp_vmf
EVOLVE_METHODS[EvolveMethod.prop_and_compress_tdrk4] = evolve_prop_and_compress_tdrk4
