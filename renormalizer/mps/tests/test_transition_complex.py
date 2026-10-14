"""The public bra argument is a ket; legacy self_conj is preconjugated."""
import numpy as np
import pytest
from renormalizer import Model, BasisHalfSpin, Mps, Mpo, Op

@pytest.mark.parametrize('opt', [False, True])
def test_complex_transition_bra_and_legacy(opt):
    model = Model([BasisHalfSpin(0), BasisHalfSpin(1)], [])
    ket = Mps.hartree_product_state(model, {0: [1, 0], 1: [1, 0]}).to_complex()
    ket[0] = ket[0].array * 1j
    bra = Mps.hartree_product_state(model, {0: [1, 0], 1: [1, 0]})
    identity = Mpo(model, Op('I', 0))
    # Independent dense inner product detects missing or double conjugation.
    expected = np.vdot(bra.todense().ravel(), ket.todense().ravel())
    assert ket.expectation(identity, bra=bra) == pytest.approx(expected)
    assert ket.expectation(identity, bra=ket) == pytest.approx(1)
    assert ket.expectation(identity, self_conj=ket.conj()) == pytest.approx(1)
    assert ket.expectations([identity], bra=ket, opt=opt)[0] == pytest.approx(1)
    assert ket.expectations([identity], self_conj=ket.conj(), opt=opt)[0] == pytest.approx(1)
