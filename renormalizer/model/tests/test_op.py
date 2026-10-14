import numpy as np
import pytest

from renormalizer.model import Op


@pytest.mark.parametrize("factor", [1 + 0j, np.complex128(1 + 0j), 2j])
def test_complex_factor_stays_complex(factor):
    # A complex factor with zero imaginary part used to reach float() and raise.
    op = Op("X", 0, factor)
    assert isinstance(op.factor, complex)
    assert op.factor == complex(factor)


@pytest.mark.parametrize("factor", [1, 0.5, np.float64(0.5), np.int64(3)])
def test_real_factor_is_python_float(factor):
    op = Op("X", 0, factor)
    assert type(op.factor) is float
    assert op.factor == float(factor)


def test_product_of_imaginary_factors():
    # 1j * -1j has zero imaginary part; the product must still be an Op.
    op = Op("X", 0, 1j) * Op("Z", 0, -1j)
    assert op.factor == 1
