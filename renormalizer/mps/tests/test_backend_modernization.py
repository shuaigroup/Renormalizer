# -*- coding: utf-8 -*-

import numpy as np
import pytest


def test_numpy_backend_gradient_capabilities_are_explicit():
    import renormalizer as r

    assert r.backend.name == "numpy"
    assert r.backend.supports_autodiff is False
    assert r.backend.supports_jit is False
    assert r.backend.supports_functional_update is True

    with pytest.raises(NotImplementedError, match="grad"):
        r.backend.transforms.grad(lambda x: x)


def test_numpy_backend_distributed_noops():
    import renormalizer as r

    x = np.array([1.0, 2.0])
    assert r.backend.rank == 0
    assert r.backend.size == 1
    assert r.backend.is_distributed is False
    assert r.backend.allreduce(x) is x
    assert r.backend.broadcast(x) is x
    assert r.backend.gather(x) == [x]
    assert r.backend.allgather(x) == [x]


def test_backend_proxy_identity_and_stale_xp_dispatch():
    import renormalizer as r
    from renormalizer.mps.backend import backend as legacy_backend
    from renormalizer.mps.backend import xp

    assert legacy_backend is r.backend
    previous = r.get_backend()
    selected = r.set_backend("numpy")
    # Names imported before the switch stay valid: the module-level proxy is
    # not replaced, it dispatches to the newly selected instance.
    from renormalizer.mps.backend import xp as reimported_xp
    assert reimported_xp is xp
    assert selected is not previous
    assert xp.current is selected

    a = xp.ones((2, 2))
    b = xp.eye(2) + (1 - xp.eye(2))
    assert xp.allclose(a, b)
    assert r.backend.name == "numpy"


def test_backend_proxy_resolves_backend_subpackage_paths():
    # ``renormalizer.backend`` is the public proxy, but dotted paths into the
    # ``renormalizer.backend`` subpackage must keep working.
    from unittest import mock
    import renormalizer as r
    import renormalizer.backend.execution as execution
    from renormalizer.backend import contracts

    assert r.backend.execution is execution
    assert r.backend.contracts is contracts
    with mock.patch("renormalizer.backend.execution.to_host") as patched:
        assert execution.to_host is patched
    # Backend attributes keep precedence over same-named submodules.
    assert r.backend.transforms is r.get_backend().transforms
    assert not hasattr(r.backend, "no_such_attribute_or_module")


def test_numpy_backend_functional_updates_return_updated_array():
    import renormalizer as r

    x = r.backend.zeros((3,))
    y = r.backend.at_set(x, 1, 2.0)
    z = r.backend.at_add(y, 1, 3.0)

    assert r.backend.numpy(y).tolist() == [0.0, 2.0, 0.0]
    assert r.backend.numpy(z).tolist() == [0.0, 5.0, 0.0]


def test_matrix_stays_host_numpy_and_asxp_uses_backend_boundary():
    from renormalizer.mps.matrix import Matrix, asnumpy, asxp
    import renormalizer as r

    mat = Matrix([[1.0, 2.0], [3.0, 4.0]])

    assert isinstance(mat.array, np.ndarray)
    assert isinstance(asnumpy(mat), np.ndarray)

    xp_array = asxp(mat)
    assert r.backend.is_array(xp_array)
    assert r.backend.numpy(xp_array).tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_asnumpy_handles_backend_array_and_list():
    from renormalizer.mps.matrix import asnumpy, asxp

    backend_array = asxp(np.array([1.0, 2.0]))
    assert asnumpy(backend_array).tolist() == [1.0, 2.0]
    assert asnumpy([1.0, 2.0]).tolist() == [1.0, 2.0]
