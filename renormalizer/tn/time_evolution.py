from typing import Union
import logging

import scipy
import opt_einsum as oe

from renormalizer.mps.backend import np
from renormalizer.lib import solve_ivp
from renormalizer.tn.tree import TensorTreeOperator, TensorTreeState, TensorTreeEnviron
from renormalizer.tn.hop_expr import hop_expr1


logger = logging.getLogger(__name__)


def time_derivative_vmf(tts: TensorTreeState, tto: TensorTreeOperator):
    # first get it right. Then add QN. Then benchmark and optimize
    environ_s = TensorTreeEnviron(tts, TensorTreeOperator.identity(tts.basis))
    environ_h = TensorTreeEnviron(tts, tto)

    deriv_list = []
    for inode, node in enumerate(tts.node_list):
        hop, _ = hop_expr1(node, tts, tto, environ_h)
        # idx1: children+physical, idx2: parent
        dim_parent = node.shape[-1]
        shape_2d = (-1, dim_parent)
        deriv = hop(node.tensor).reshape(shape_2d)
        if node.parent is not None:
            # apply projector and S^-1
            tensor = node.tensor.reshape(shape_2d)
            proj = tensor.conj() @ tensor.T
            ovlp = environ_s.node_list[inode].environ_parent.reshape(dim_parent, dim_parent)
            ovlp_inv = regularized_inversion(ovlp, 1e-10)
            deriv = oe.contract("bf, bg, fh -> gh", deriv, np.eye(proj.shape[0]) - proj, ovlp_inv.T)
        qnmask = tts.get_qnmask(node).reshape(deriv.shape)
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


def evolve(tts:TensorTreeState, tto:TensorTreeOperator, tau:Union[complex, float], first_step=None):
    imag_time = np.iscomplex(tau)
    # trick to avoid complex algebra
    # exp{coeff * H * tau}
    # coef and tau are different from MPS implementation
    if imag_time:
        coef = 1
        tau = tau.imag
    else:
        coef = -1j
        tts = tts.to_complex()

    def ivp_func(t, params):
        tts_t = TensorTreeState.from_tensors(tts, params)
        return coef * time_derivative_vmf(tts_t, tto)
    init_y = np.concatenate([node.tensor[tts.get_qnmask(node)].ravel() for node in tts.node_list])
    sol = solve_ivp(ivp_func, (0, tau), init_y, first_step=first_step, rtol=1e-4, atol=1e-7)
    logger.info(f"VMF func called: {sol.nfev}. RKF steps: {len(sol.t)}")
    new_tts = TensorTreeState.from_tensors(tts, sol.y[:, -1])
    new_tts.canonicalise()
    return new_tts
