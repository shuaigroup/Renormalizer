"""Small dense matvec oracle for both variational sweep directions."""
import numpy as np
import pytest
from renormalizer import BasisHalfSpin, Model, Mpo, Mps, Op
from renormalizer.mps import MpDm
from renormalizer.mps.mp import MatrixProduct
from renormalizer.mps.matrix import asnumpy
from renormalizer.utils import CompressConfig, CompressCriteria


@pytest.mark.parametrize('family', ['mps', 'mpdm', 'mpo'])
@pytest.mark.parametrize('complex_values', [False, True])
@pytest.mark.parametrize('method', ['1site', '2site'])
def test_variational_compression_dense_both_directions(captured_backend, monkeypatch, family, complex_values, method):
    model = Model([BasisHalfSpin(i) for i in range(3)], [Op('sigma_z', 0), Op('sigma_z sigma_z', [1, 2], .3)])
    operator = Mpo(model)
    if family == 'mpo':
        state = Mpo(model)
    else:
        state = Mps.random(model, 0, 4)
        if family == 'mpdm':
            state = MpDm.from_mps(state)
    if complex_values:
        state = state.to_complex().scale(.6 + .8j)
        operator = operator.scale(-1j)
    dense_before = asnumpy(state.todense()).copy()
    signs = np.array([1., -1.])
    h = np.diag(np.kron(np.kron(signs, np.ones(2)), np.ones(2)) + .3 * np.kron(np.ones(2), np.kron(signs, signs)))
    if complex_values:
        h = -1j * h
    expected = h @ dense_before.reshape(8, -1)
    state.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=8)
    state.compress_config.vmethod = method
    state.compress_config.vprocedure = [[8, 0.]] * 4
    directions = set()
    original = MatrixProduct._update_mps
    def update(self, *args, **kwargs):
        directions.add(self.to_right)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(MatrixProduct, '_update_mps', update)
    actual = state.variational_compress(operator)
    assert directions == {False, True}
    np.testing.assert_allclose(asnumpy(actual.todense()).reshape(8, -1), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(asnumpy(state.todense()), dense_before)
