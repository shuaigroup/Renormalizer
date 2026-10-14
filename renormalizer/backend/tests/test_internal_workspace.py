"""Internal owned workspace writes; distinct from public functional updates."""
import os
import numpy as np
import pytest
from renormalizer.backend.context import make_context


def assert_native(adapter, value, dtype=None):
    # Legacy NumPy has no device ownership/dtype wrapper; arrays carry both.
    if adapter.name == 'numpy':
        assert isinstance(value, np.ndarray)
        actual_dtype = value.dtype
    else:
        assert adapter.owns(value)
        actual_dtype = adapter.dtype_of(value)
    if dtype is not None:
        assert actual_dtype == np.dtype(dtype)


@pytest.fixture
def context():
    return make_context(os.environ.get('RENO_TEST_BACKEND', 'numpy'),
                        device=os.environ.get('RENO_TEST_DEVICE', 'cpu'),
                        real_dtype='float64')


def test_owned_write_retains_mutable_storage_and_functional_api(context):
    adapter = context.adapter
    original = np.arange(12., dtype=np.float64).reshape(3, 4)
    workspace = context.ops.from_numpy(original, copy=True)
    functional = context.ops.at_set(workspace, (1, slice(None)), context.ops.array([9.] * 4))
    np.testing.assert_array_equal(context.ops.to_numpy(workspace), original)
    expected = original.copy()
    expected[1] = 9
    np.testing.assert_array_equal(context.ops.to_numpy(functional), expected)
    result = adapter.write_owned(workspace, (1, slice(None)), context.ops.array([9.] * 4))
    assert_native(adapter, result)
    if adapter.name == 'jax':
        np.testing.assert_array_equal(context.ops.to_numpy(workspace), original)
    else:
        assert result is workspace
    np.testing.assert_array_equal(context.ops.to_numpy(result), expected)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
def test_native_identity_and_broadcast_copy(context, dtype):
    adapter = context.adapter
    eye = adapter.identity(3, dtype=np.dtype(dtype))
    assert_native(adapter, eye, dtype)
    np.testing.assert_array_equal(context.ops.to_numpy(eye), np.eye(3, dtype=dtype))
    row = context.ops.array([1, 2, 3], dtype=dtype)
    expanded = adapter.array(adapter.broadcast_to(row, (2, 3)), copy=True)
    updated = adapter.write_owned(expanded, (0, 0), 7)
    np.testing.assert_array_equal(context.ops.to_numpy(row), [1, 2, 3])
    np.testing.assert_array_equal(context.ops.to_numpy(updated), [[7, 2, 3], [1, 2, 3]])


def test_torch_identity_default_tracks_adapter_precision(context):
    if context.adapter.name != 'torch':
        pytest.skip('Torch-specific adapter precision default')
    adapter = make_context('torch', device=context.device, real_dtype='float32').adapter
    out = adapter.identity(0)
    assert adapter.owns(out)
    assert adapter.dtype_of(out) == np.dtype('float32')
    assert tuple(out.shape) == (0, 0)


def test_environment_storage_follows_backend_for_mps_and_ttns(context):
    # Environments kept across sweep steps are native arrays when the backend
    # keeps workspaces native (no upload per read), host arrays otherwise; the
    # same policy applies to MPS and TTNS.
    from renormalizer.backend.context import capture_backend
    from renormalizer.backend.tests.spin_reference import spin_fixture, host_seed
    from renormalizer.mps import Mps, Mpo
    from renormalizer.mps.lib import Environ
    from renormalizer.tn import BasisTree, TTNO, TTNS
    from renormalizer.tn.tree import TTNEnviron
    adapter = context.adapter
    model, _ = spin_fixture(4, 0.137)
    with host_seed():
        mps, mpo = Mps.random(model, 0, 4), Mpo(model)
        tree = BasisTree.binary(model.basis)
        ttns, ttno = TTNS.random(tree, qntot=0, m_max=4), TTNO(tree, model.ham_terms)

    def check(value):
        if adapter.native_workspace_storage:
            assert adapter.owns(value)
        else:
            assert isinstance(value, np.ndarray)

    with capture_backend(adapter):
        environ = Environ(mps, mpo, "R")
        ttne = TTNEnviron(ttns, ttno)
    for (domain, idx), value in environ._virtual_disk.items():
        if 0 <= idx < len(mps):
            check(value)
    for enode in ttne.node_list:
        for value in enode.environ_children:
            check(value)
        if enode.parent is not None:
            check(enode.environ_parent)
