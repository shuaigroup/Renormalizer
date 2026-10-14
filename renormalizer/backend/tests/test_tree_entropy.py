"""Analytic Schmidt entropy without changing historical native RDM outputs."""
import numpy as np
import pytest
from renormalizer import BasisHalfSpin
from renormalizer.tn import BasisTree, TTNS


@pytest.mark.parametrize('operation', ['one_site', 'one_dof', 'two_site', 'two_dof', 'mutual'])
def test_native_tree_rdm_entropy_host_solver(captured_backend, operation):
    tree = BasisTree.binary([BasisHalfSpin(0), BasisHalfSpin(1)])
    state = TTNS(tree, {})
    p = np.array([.25, .75])
    child, = state.root.children
    state.root.tensor = np.diag(np.sqrt(p)).reshape(2, 2, 1)
    child.tensor = np.eye(2)
    child.qn = np.zeros((2, 1), dtype=int)
    state.check_shape()
    expected = -np.sum(p * np.log(p))
    # Public RDM arrays keep their existing backend-native boundary.
    for dm in state.calc_1dof_rdm().values():
        if captured_backend.adapter.name != 'numpy':
            assert captured_backend.adapter.owns(dm)
    if operation == 'one_site':
        actual = list(state.calc_1site_entropy().values())
    elif operation == 'one_dof':
        actual = list(state.calc_1dof_entropy().values())
    elif operation == 'two_site':
        actual = state.calc_2site_entropy((0, 1))[(0, 1)]
        expected = 0.
    elif operation == 'two_dof':
        actual = state.calc_2dof_entropy((0, 1))[(0, 1)]
        expected = 0.
    else:
        info, _ = state.calc_2dof_mutual_info((0, 1))
        actual = info[(0, 1)]  # documented half-mutual-information convention
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
