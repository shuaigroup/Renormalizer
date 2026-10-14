"""Purification RDM oracle, physical-row/ancilla-column dense amplitude."""
import itertools
import numpy as np
from renormalizer import BasisHalfSpin, Model
from renormalizer.mps import MpDm
from renormalizer.mps.matrix import asnumpy


def test_complex_purification_one_and_two_site_partial_traces(captured_backend):
    model = Model([BasisHalfSpin(i) for i in range(3)], [])
    amplitude = (np.arange(1, 65) + 1j * np.arange(64, 0, -1)).reshape(8, 8)
    amplitude /= np.linalg.norm(amplitude)
    # Physical sites followed by ancillas -> paired local physical/ancilla axes.
    residual = amplitude.reshape((2,) * 6).transpose(0, 3, 1, 4, 2, 5).reshape(1, 4, 4, 4)
    state = MpDm(); state.model = model; state.dtype = np.complex128
    for _ in range(2):
        left = residual.shape[0]
        q, r = np.linalg.qr(residual.reshape(left * 4, -1))
        state.append(q.reshape(left, 2, 2, q.shape[1]))
        residual = r.reshape((r.shape[0],) + residual.shape[2:])
    state.append(residual.reshape(residual.shape[0], 2, 2, 1))
    state.build_empty_qn()
    np.testing.assert_allclose(asnumpy(state.todense()), amplitude, atol=1e-12)
    density = amplitude @ amplitude.conj().T
    def partial_trace(keep):
        other = [i for i in range(3) if i not in keep]
        axes = list(keep) + other + [i + 3 for i in keep] + [i + 3 for i in other]
        reduced = density.reshape((2,) * 6).transpose(axes).reshape(2**len(keep), 2**len(other), 2**len(keep), 2**len(other))
        return np.einsum('abcb->ac', reduced)
    one = state.calc_1site_rdm(); two = state.calc_2site_rdm()
    for i in range(3):
        np.testing.assert_allclose(asnumpy(one[i]), partial_trace((i,)), atol=1e-12)
    for pair in itertools.combinations(range(3), 2):
        np.testing.assert_allclose(asnumpy(two[pair]), partial_trace(pair), atol=1e-12)
