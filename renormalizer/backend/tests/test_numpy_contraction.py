"""The cached NumPy tensordot must be indistinguishable from numpy.tensordot."""
import numpy as np
import opt_einsum
import pytest

from renormalizer.backend import numpy_contraction
from renormalizer.backend.numpy_backend import NumpyBackend


def _operands(rng, shape_a, shape_b, dtype_a=np.float64, dtype_b=np.float64):
    def draw(shape, dtype):
        x = rng.standard_normal(shape)
        if np.issubdtype(dtype, np.complexfloating):
            x = x + 1j * rng.standard_normal(shape)
        return x.astype(dtype)
    return draw(shape_a, dtype_a), draw(shape_b, dtype_b)


CASES = [
    ((4, 3, 4), (4, 3, 4), 1),
    ((4, 3, 4), (4, 3, 4), 0),
    ((2, 3, 4), (3, 4, 5), 2),
    ((16, 2, 16), (16, 2, 16), ([2], [0])),
    ((16, 2, 16), (16, 2, 16), (-1, 0)),
    ((16, 2, 16), (2, 16, 2, 16), ([1, 2], [0, 1])),
    ((5, 4, 3, 2), (3, 4, 5), ([0, 1, 2], [2, 1, 0])),
    ((5, 4, 3, 2), (3, 4, 5), ([-4, -3, -2], [-1, 1, 0])),
    ((3, 0, 4), (4, 2), ([2], [0])),
    ((3, 4), (4,), ([1], [0])),
    ((2, 2, 2, 2), (2, 2, 2, 2), (range(2, 4), range(2, 4))),
    ((2, 2, 2, 2), (2, 2, 2, 2), np.int64(2)),
]


@pytest.mark.parametrize('shape_a,shape_b,axes', CASES)
@pytest.mark.parametrize('dtypes', [(np.float64, np.float64), (np.complex128, np.float64),
                                    (np.float32, np.complex64)])
def test_bitwise_identical_to_numpy(shape_a, shape_b, axes, dtypes):
    a, b = _operands(np.random.default_rng(7), shape_a, shape_b, *dtypes)
    expected = np.tensordot(a, b, axes)
    for _ in range(2):  # planned on the first call, cached on the second
        result = numpy_contraction.tensordot(a, b, axes)
        assert result.dtype == expected.dtype and result.shape == expected.shape
        assert np.array_equal(result, expected, equal_nan=True)


def test_non_contiguous_operands():
    a, b = _operands(np.random.default_rng(3), (8, 6, 10), (10, 6, 4))
    a, b = a[::2, :, 1:], np.asfortranarray(b)[1:, ::-1]
    axes = ([1, 2], [1, 0])
    assert np.array_equal(numpy_contraction.tensordot(a, b, axes), np.tensordot(a, b, axes))


@pytest.mark.parametrize('axes', [([0], [0]), ([3], [0]), ([0, 1], [0]), (iter([2]), iter([0]))])
def test_unsupported_or_invalid_axes_behave_like_numpy(axes):
    a, b = _operands(np.random.default_rng(5), (2, 3, 4), (4, 5))
    try:
        expected = np.tensordot(a, b, axes)
    except Exception as error:
        with pytest.raises(type(error)):
            numpy_contraction.tensordot(a, b, axes)
    else:
        assert np.array_equal(numpy_contraction.tensordot(a, b, axes), expected)


def test_non_ndarray_operands_go_to_numpy():
    a = [[1.0, 2.0], [3.0, 4.0]]
    assert np.array_equal(numpy_contraction.tensordot(a, a, 1), np.tensordot(a, a, 1))


def test_numpy_adapter_uses_it_for_tensordot_and_opt_einsum():
    adapter = NumpyBackend()
    assert adapter.tensordot is numpy_contraction.tensordot
    # Public name unchanged; only Renormalizer's own contractions use the module.
    assert adapter.opt_einsum_name == 'numpy'
    assert adapter.opt_einsum_module == numpy_contraction.__name__
    a, b, c = (np.random.default_rng(i).standard_normal((6, 6)) for i in range(3))
    expected = opt_einsum.contract('ij,jk,kl->il', a, b, c, backend='numpy')
    result = opt_einsum.contract('ij,jk,kl->il', a, b, c, backend=adapter.opt_einsum_module)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize('strict', [False, True])
def test_non_integer_axes(monkeypatch, strict):
    a, b = _operands(np.random.default_rng(6), (2, 3), (3, 4))
    if strict:
        monkeypatch.setattr(numpy_contraction, '_axes_key', numpy_contraction._strict_axes_key)
    numpy_contraction._plan.cache_clear()
    for axes in (([1.0], [0]), (1.0, 0), 1.0):
        with pytest.raises(TypeError):
            np.tensordot(a, b, axes)
        with pytest.raises(TypeError):  # no integer plan cached yet: rejected in both modes
            numpy_contraction.tensordot(a, b, axes)
    numpy_contraction.tensordot(a, b, ([1], [0]))  # plan cached for the integer axes
    if strict:
        with pytest.raises(TypeError):
            numpy_contraction.tensordot(a, b, ([1.0], [0]))
    else:
        # Documented default: equal non-integer axes reuse the integer plan.
        assert np.array_equal(numpy_contraction.tensordot(a, b, ([1.0], [0])), np.tensordot(a, b, ([1], [0])))


def test_proxy_tensordot_resolves_to_the_cached_tensordot():
    from renormalizer.backend.context import capture_backend, internal_backend as xp
    with capture_backend(NumpyBackend()):
        assert xp.tensordot is numpy_contraction.tensordot
