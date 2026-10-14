from types import SimpleNamespace

import numpy as np
import pytest

from renormalizer.backend.context import capture_backend
from renormalizer.backend.numpy_backend import NumpyBackend
from renormalizer.mps.block_env import supports_block_env, BlockEnvData


@pytest.mark.parametrize('name', ['cupy', 'torch', 'jax'])
def test_block_env_uses_captured_backend(name):
    ms = np.ones((1, 2, 1))
    mo = np.ones((1, 2, 2, 1))
    # Only dispatch metadata is needed: this test does not require GPU libraries.
    with capture_backend(SimpleNamespace(name=name, numpy=np.asarray)):
        assert not supports_block_env(ms, mo)
        with capture_backend(NumpyBackend()):
            assert supports_block_env(ms, mo)
        assert not supports_block_env(ms, mo)


def test_numpy_context_builds_real_block_environments():
    from renormalizer import BasisHalfSpin, Model, Mps, Mpo, Op
    from renormalizer.mps.lib import Environ

    with capture_backend(NumpyBackend()):
        model = Model([BasisHalfSpin(i) for i in range(3)], [Op('Z', 0)])
        state = Mps.hartree_product_state(model, condition={})
        operator = Mpo(model)
        dense = Environ(state, operator)
        blocks = Environ(state, operator, use_block_env=True, block_env_min_bond_dim=0)
        raw = blocks.read_raw('L', 0)
        assert isinstance(raw, BlockEnvData)
        np.testing.assert_allclose(raw.to_dense(), dense.read('L', 0))
