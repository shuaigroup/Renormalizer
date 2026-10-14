"""Interactions covered independently of the individual adapter smoke tests."""
import numpy as np
import pytest

from renormalizer.lib.integrate.integrate import solve_ivp
from renormalizer.mps.matrix import Matrix
from renormalizer.tn.node import TreeNodeTensor


def test_backward_complex_terminal_dense_output(captured_backend):
    ctx = captured_backend
    initial = np.array([1 + 2j, 2 - 1j])

    def event(t, y):
        # Event time is control metadata; the evolving state stays native.
        return t - .045

    event.terminal = True
    event.direction = -1
    result = solve_ivp(
        lambda t, y: -y, (.1, 0.), ctx.ops.from_numpy(initial),
        rtol=1e-9, atol=1e-11, t_eval=np.linspace(.1, 0., 11),
        dense_output=True, events=event,
    )

    def host(value):
        # Check the actual candidate before downloading for the analytic oracle.
        if ctx.adapter.name == 'numpy':
            assert isinstance(value, np.ndarray)
        else:
            assert ctx.adapter.owns(value)
        return np.asarray(ctx.adapter.numpy(value))

    assert result.success and result.status == 1
    times = host(result.t)
    np.testing.assert_allclose(times, [.1, .09, .08, .07, .06, .05], atol=1e-14)
    np.testing.assert_allclose(
        host(result.y), initial[:, None] * np.exp(-(times - .1)),
        rtol=1e-8, atol=1e-9,
    )
    # Repeated unsorted queries include the terminal root between sample times.
    queries = np.array([.045, .095, .06, .045])
    np.testing.assert_allclose(
        host(result.sol(queries)), initial[:, None] * np.exp(-(queries - .1)),
        rtol=1e-8, atol=1e-9,
    )
    np.testing.assert_allclose(host(result.t_events[0]), [.045], atol=1e-12)


@pytest.mark.parametrize('kind', ['matrix', 'tree'])
@pytest.mark.parametrize('writable', [True, False], ids=['writable_alias', 'readonly_copy'])
def test_persistent_storage_alias_contract(captured_backend, kind, writable):
    source = np.ones((1, 2, 1), dtype=np.float64)
    source.flags.writeable = writable
    obj = Matrix(source) if kind == 'matrix' else TreeNodeTensor(source)
    stored = obj.array if kind == 'matrix' else obj.tensor
    # Existing writable host inputs retain aliasing; read-only exports need
    # independent storage so subsequent tensor updates can safely mutate it.
    assert stored.flags.writeable
    assert np.shares_memory(source, stored) == writable
    stored[...] = 3.
    np.testing.assert_array_equal(source, np.full(source.shape, 3. if writable else 1.))
    np.testing.assert_array_equal(stored, np.full(source.shape, 3.))
