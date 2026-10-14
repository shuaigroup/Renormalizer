import numpy as np
import pytest

from renormalizer import BasisHalfSpin
from renormalizer.tn.tree import TTNS, TTNO, TTNEnviron
from renormalizer.tn.treebase import BasisTree


def test_self_bra_and_overlap_leave_roots_detached():
    basis = BasisTree.binary([BasisHalfSpin(i) for i in range(3)])
    state = TTNS.random(basis, qntot=0, m_max=2).scale(1 + 0.2j)
    operator = TTNO.identity(basis)
    expected = np.vdot(state.todense(), state.todense())
    assert state.expectation(operator, bra=state) == pytest.approx(expected)
    assert state.overlap(state) == pytest.approx(expected)
    assert state.root.parent is None
    assert operator.root.parent is None
    assert basis.root.parent is None


@pytest.mark.parametrize('distinct_bra', [False, True])
def test_failed_expectation_restores_roots(monkeypatch, distinct_bra):
    basis = BasisTree.binary([BasisHalfSpin(i) for i in range(3)])
    state = TTNS.random(basis, qntot=0, m_max=2)
    bra = TTNS.random(basis, qntot=0, m_max=2) if distinct_bra else None
    operator = TTNO.identity(basis)

    def fail(*args, **kwargs):
        raise RuntimeError('injected contraction failure')

    with monkeypatch.context() as patch:
        patch.setattr(TTNEnviron, 'build_children_environ', fail)
        with pytest.raises(RuntimeError, match='injected contraction failure'):
            state.expectation(operator, bra=bra)
    assert state.root.parent is None
    assert operator.root.parent is None
    assert basis.root.parent is None
    if bra is not None:
        assert bra.root.parent is None
    # A failed calculation must not poison subsequent use of either tree.
    expected_bra = state if bra is None else bra
    expected = np.vdot(expected_bra.todense(), state.todense())
    assert state.expectation(operator, bra=bra) == pytest.approx(expected)
