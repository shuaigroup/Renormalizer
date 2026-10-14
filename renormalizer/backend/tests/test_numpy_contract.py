import numpy as np
import pytest

from renormalizer.backend.numpy_backend import NumpyBackend


@pytest.mark.parametrize('x', [np.arange(8.)[::2], np.arange(6.)[::-1],
                               np.zeros((0, 3)), np.array(2.)])
def test_copy_false_preserves_existing_storage(x):
    out = NumpyBackend().array(x, copy=False)
    assert out is x


@pytest.mark.parametrize('data,dtype', [([1., 2.], None), (np.ones(2), np.float32)])
def test_copy_false_rejects_required_allocation(data, dtype):
    with pytest.raises(ValueError, match='copy=False'):
        NumpyBackend().array(data, dtype=dtype, copy=False)


def test_copy_true_independent_and_none_allows_conversion():
    b = NumpyBackend()
    x = np.arange(6.).reshape(2, 3)
    y = b.array(x, copy=True)
    assert not np.shares_memory(x, y)
    y[0, 0] = 99
    assert x[0, 0] == 0
    assert b.array([1, 2], copy=None).tolist() == [1, 2]
    assert b.array(x, dtype=np.float32, copy=None).dtype == np.float32


def test_host_boundaries_and_legacy_none():
    b = NumpyBackend()
    x = np.arange(4.)
    assert b.from_numpy(x, copy=False) is x
    assert b.to_numpy(x, copy=False) is x
    assert not np.shares_memory(b.to_numpy(x, copy=True), x)
    assert b.numpy(None) is None


def test_copy_false_order_ndmin_and_invalid_policy():
    b = NumpyBackend()
    x = np.arange(8.)[::2]
    with pytest.raises(ValueError, match='copy=False'):
        b.array(x, order='C', copy=False)
    y = b.array(x, ndmin=2, copy=False)
    assert y.shape == (1, 4) and np.shares_memory(x, y)
    with pytest.raises(TypeError, match='copy'):
        b.array(x, copy='yes')


def test_copy_false_subclass_order_and_empty_view():
    b = NumpyBackend()
    class Subclass(np.ndarray):
        pass
    original = np.arange(8.).view(Subclass)
    same = b.array(original, copy=False, subok=True)
    assert same is original
    plain = b.array(original, copy=False, subok=False)
    assert type(plain) is np.ndarray and np.shares_memory(plain, original)
    empty = original[:0]
    assert b.array(empty, copy=False, subok=True) is empty
    with pytest.raises(ValueError):
        b.array(original, copy=False, order='bad')
    with pytest.raises(TypeError):
        b.array(original, copy=False, unsupported_option=True)


def test_legacy_array_default_copies_but_explicit_none_may_alias():
    b = NumpyBackend()
    source = np.arange(4.)
    copied = b.array(source)
    copied[0] = 99
    assert source[0] == 0
    assert b.array(source, copy=None) is source


@pytest.mark.parametrize('order', ['A', 'C', 'F'])
def test_copy_false_contiguous_orders_reject_strided_vector(order):
    with pytest.raises(ValueError, match='copy=False'):
        NumpyBackend().array(np.arange(8.)[::2], order=order, copy=False)
