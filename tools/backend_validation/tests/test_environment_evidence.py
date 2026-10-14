from tools.backend_validation.environment import validate_identity
import numpy as np
from tools.backend_validation import environment


def test_actual_numpy_major_and_backend_are_required():
    assert validate_identity('numpy2', '1.26.4', 'numpy')['reason'] == 'numpy_major_mismatch'
    assert validate_identity('jax', '1.26.4', 'numpy')['reason'] == 'backend_mismatch'
    assert validate_identity('numpy2', '2.0.0', 'numpy')['ok']
    assert not validate_identity('invented', '2.0.0', 'numpy')['ok']


def test_cpu_smoke_checks_native_matmul_and_svd():
    report = environment.collect('numpy' + np.__version__.split('.')[0], device='cpu')
    assert report['ok'], report
    assert report['device'] == 'cpu'
    assert report['dtype'] == 'float64'
    assert report['checks']['matmul']['ok']
    assert report['checks']['svd_reconstruction']['ok']


def test_requested_gpu_never_becomes_numpy_cpu():
    report = environment.collect('numpy' + np.__version__.split('.')[0], device='cuda:0')
    assert not report['ok']
    assert report['requested_device'] == 'cuda:0'


def test_witness_uses_native_object_and_numeric_failure_is_visible():
    evidence = environment.array_evidence(np.ones((2, 2)))
    assert evidence['backend'] == 'numpy' and evidence['device'] == 'cpu'
    assert not environment.check_error(np.zeros((2, 2)), np.ones((2, 2)))['ok']


def test_smoke_rejects_wrong_native_device(monkeypatch):
    original = environment.array_evidence
    def wrong_device(array):
        return {**original(array), 'device': 'cuda:0'}
    monkeypatch.setattr(environment, 'array_evidence', wrong_device)
    report = environment.collect('numpy' + np.__version__.split('.')[0], device='cpu')
    assert not report['ok'] and report['reason'] == 'native_device_mismatch'
