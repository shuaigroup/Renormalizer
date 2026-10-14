"""Cached transitions retain bilinear contraction and native dtype promotion."""
import numpy as np
import pytest
from renormalizer import BasisHalfSpin, Model, Mpo, Mps, Op


@pytest.mark.parametrize('optimized', [False, True])
def test_cached_complex_transition_matches_dense(captured_backend, optimized):
    model = Model([BasisHalfSpin(0), BasisHalfSpin(1)], [])
    real = Mps.hartree_product_state(model, {}).to_complex()
    ket = real.copy()
    ket[0] = ket[0].array * 1j
    operators = [Mpo(model, Op('I', 0)), Mpo(model, Op('sigma_z', 1))]
    for bra in [real, ket]:
        expected = np.array([np.vdot(bra.todense().ravel(), op.todense() @ ket.todense().ravel()) for op in operators])
        actual = ket.expectations(operators, bra=bra, opt=optimized)
        legacy = ket.expectations(operators, self_conj=bra.conj(), opt=optimized)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(legacy, expected, atol=1e-12)
