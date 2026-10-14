import numpy as np
from tools.backend_validation.fixtures import matmul_fixture


def test_fixture_content_and_independent_reference():
    a, b, expected = matmul_fixture(27)
    a2, b2, expected2 = matmul_fixture(27)
    np.testing.assert_array_equal(a, a2)
    np.testing.assert_array_equal(b, b2)
    np.testing.assert_array_equal(expected, expected2)
    np.testing.assert_allclose(a @ b, expected, atol=1e-12, rtol=1e-12)
    assert a.shape == (7, 5) and b.shape == (5, 3)


def test_manifest_freezes_values_and_tolerances(tmp_path):
    import hashlib
    import json
    from tools.backend_validation.fixtures import write_fixture
    manifest_path = write_fixture(tmp_path, seed=27)
    manifest = json.loads(manifest_path.read_text())
    assert manifest['schema_version'] == 1
    assert manifest['op'] == 'matmul'
    assert manifest['tolerances']['atol_F'] == 1e-12
    for entry in manifest['arrays'].values():
        path = tmp_path / entry['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        array = np.load(path, allow_pickle=False)
        assert list(array.shape) == entry['shape']
        assert str(array.dtype) == entry['dtype']
