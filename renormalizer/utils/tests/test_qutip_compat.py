"""Independent small-state controls for the supported QuTiP public API."""
import numpy as np

from renormalizer.utils.qutip_utils import get_gs


def test_ground_state_dimensions_and_values():
    state = get_gs(2, 3)
    expected = np.zeros((36, 1), dtype=complex)
    expected[0] = 1
    np.testing.assert_array_equal(state.full(), expected)
    assert abs(state.norm() - 1) < 1e-14
