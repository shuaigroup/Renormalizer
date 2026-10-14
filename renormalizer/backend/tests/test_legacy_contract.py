import importlib
import random

import numpy as np
import pytest

from renormalizer.backend.numpy_backend import NumpyBackend
from renormalizer.backend.proxy import BackendManager, BackendProxy


def test_old_names_and_fp32_tolerance():
    legacy = importlib.import_module('renormalizer.mps.backend')
    for name in ('GPU_KEY', 'GPU_ID', 'try_import_cupy', 'xpseed',
                 'npseed', 'randomseed', 'Backend'):
        assert hasattr(legacy, name), name
    b = NumpyBackend()
    b.use_32bits()
    assert b.canonical_rtol == 1e-2
    with pytest.raises(RuntimeError, match='only be initialized once'):
        legacy.Backend()


def test_cached_proxy_and_captured_instance(monkeypatch):
    manager = BackendManager('numpy')
    cached = BackendProxy(manager)
    first = manager.get_backend()
    second = NumpyBackend()
    monkeypatch.setattr(importlib.import_module('renormalizer.backend.proxy'), 'create_backend',
                        lambda *args, **kwargs: second)
    assert manager.set_backend('numpy') is second
    assert cached.current is second
    assert first is not second


def test_failed_seed_does_not_publish_or_change_numpy_rng(monkeypatch):
    import renormalizer.cons as cons
    before = cons.get_backend()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    class BadRandom:
        def seed(self, value):
            np.random.seed(17)
            random.seed(18)
            raise RuntimeError('seed failure')
    candidate = NumpyBackend()
    candidate.random = BadRandom()
    monkeypatch.setattr(importlib.import_module('renormalizer.backend.proxy'), 'create_backend',
                        lambda *args, **kwargs: candidate)
    try:
        with pytest.raises(RuntimeError, match='seed failure'):
            cons.set_backend('numpy')
        assert cons.get_backend() is before
        actual = np.random.get_state()
        assert actual[0] == numpy_state[0]
        np.testing.assert_array_equal(actual[1], numpy_state[1])
        assert actual[2:] == numpy_state[2:]
        assert random.getstate() == python_state
    finally:
        cons._manager.current = before
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def test_failed_creation_does_not_publish(monkeypatch):
    manager = BackendManager('numpy')
    before = manager.current
    def fail(*args, **kwargs):
        raise ImportError('optional backend absent')
    monkeypatch.setattr(importlib.import_module('renormalizer.backend.proxy'), 'create_backend', fail)
    with pytest.raises(ImportError):
        manager.set_backend('cupy')
    assert manager.current is before


def test_precision_change_is_transactional():
    b = NumpyBackend()
    b.first_mp = True
    old = b.dtypes
    with pytest.raises(RuntimeError):
        b.dtypes = (np.float32, np.complex64)
    assert b.dtypes == old


def test_legacy_auto_probe_and_explicit_cpu_precedence(monkeypatch):
    import sys
    from types import SimpleNamespace
    from renormalizer.backend.factory import create_backend
    devices = []
    class Device:
        def __init__(self, value):
            devices.append(value)
        def use(self):
            pass
    fake_cupy = SimpleNamespace(cuda=SimpleNamespace(Device=Device,
                    runtime=SimpleNamespace(CUDARuntimeError=RuntimeError)))
    # The probe imports CuPy, but adapter construction is isolated from real CUDA.
    class FakeCupyBackend:
        name = 'cupy'
    monkeypatch.setitem(sys.modules, 'cupy', fake_cupy)
    monkeypatch.setitem(sys.modules, 'renormalizer.backend.cupy_backend',
                        SimpleNamespace(CupyBackend=FakeCupyBackend))
    monkeypatch.delenv('RENO_GPU', raising=False)
    assert create_backend(None, explicit=False).name == 'cupy'
    assert devices == [0]
    monkeypatch.setenv('RENO_GPU', '2')
    assert create_backend(None, explicit=False).name == 'cupy'
    assert devices[-1] == '2'
    count = len(devices)
    assert create_backend('numpy', explicit=True).name == 'numpy'
    assert len(devices) == count


def test_legacy_probe_failure_returns_numpy(monkeypatch):
    import sys
    from types import SimpleNamespace
    from renormalizer.backend.factory import create_backend
    class CudaError(RuntimeError):
        pass
    class Device:
        def __init__(self, value):
            pass
        def use(self):
            raise CudaError('device unavailable')
    fake_cupy = SimpleNamespace(cuda=SimpleNamespace(Device=Device,
                    runtime=SimpleNamespace(CUDARuntimeError=CudaError)))
    monkeypatch.setitem(sys.modules, 'cupy', fake_cupy)
    assert create_backend(None, explicit=False).name == 'numpy'


@pytest.mark.parametrize('value', ['', '0', '1'])
def test_fp32_environment_is_presence_based(monkeypatch, value):
    monkeypatch.setenv('RENO_FP32', value)
    assert NumpyBackend().real_dtype == np.float32


def test_snapshot_query_follows_selection_but_old_values_stay_snapshots():
    import renormalizer.cons as cons
    legacy = importlib.import_module('renormalizer.mps.backend')
    old = (legacy.USE_GPU, legacy.OE_BACKEND, legacy.MEMORY_ERRORS, legacy.ARRAY_TYPES)
    original = cons.get_backend()
    state = np.random.get_state()
    try:
        selected = cons.set_backend('numpy')
        snapshot = legacy.backend_snapshot()
        assert snapshot['USE_GPU'] is False
        assert snapshot['OE_BACKEND'] == 'numpy'
        assert snapshot['MEMORY_ERRORS'] == selected.memory_errors
        assert snapshot['ARRAY_TYPES'] == (np.ndarray,)
        assert (legacy.USE_GPU, legacy.OE_BACKEND, legacy.MEMORY_ERRORS, legacy.ARRAY_TYPES) == old
        assert legacy.backend is cons.backend and legacy.xp is cons.backend
        np.testing.assert_array_equal(np.random.random(4), np.random.RandomState(2019).random(4))
    finally:
        cons._manager.current = original
        np.random.set_state(state)


def test_old_probe_function_updates_default_gpu_id(monkeypatch):
    import sys
    from types import SimpleNamespace
    legacy = importlib.import_module('renormalizer.mps.backend')
    devices = []
    class Device:
        def __init__(self, value):
            devices.append(value)
        def use(self):
            pass
    fake_cupy = SimpleNamespace(cuda=SimpleNamespace(Device=Device,
                    runtime=SimpleNamespace(CUDARuntimeError=RuntimeError)))
    monkeypatch.setitem(sys.modules, 'cupy', fake_cupy)
    monkeypatch.setattr(legacy, 'GPU_ID', None, raising=False)
    use_gpu, namespace = legacy.try_import_cupy()
    assert use_gpu is True and namespace is fake_cupy
    assert legacy.GPU_ID == 0 and devices == [0]


@pytest.mark.parametrize('fp32', [None, '', '0', '1'])
def test_import_seeds_environment_and_missing_cupy_subprocess(fp32):
    import json
    import os
    import subprocess
    import sys
    script = '''
import importlib.abc, importlib, json, sys
class NoCupy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cupy' or fullname.startswith('cupy.'):
            raise ImportError('CuPy deliberately absent')
sys.meta_path.insert(0, NoCupy())
import renormalizer
import numpy as np
import random
legacy = importlib.import_module('renormalizer.mps.backend')
print(json.dumps({'backend': legacy.backend.name,
    'fp32': legacy.backend.is_32bits, 'gpu_id': legacy.GPU_ID,
    'numpy_draw': float(np.random.random()), 'python_draw': random.random(),
    'seeds': [legacy.xpseed, legacy.npseed, legacy.randomseed]}))
'''
    env = os.environ.copy()
    env['RENO_GPU'] = '17'
    env.pop('RENO_FP32', None)
    if fp32 is not None:
        env['RENO_FP32'] = fp32
    result = subprocess.run([sys.executable, '-c', script], env=env,
                            capture_output=True, text=True, check=True, timeout=30)
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {'backend': 'numpy', 'fp32': fp32 is not None,
        'gpu_id': '17', 'numpy_draw': float(np.random.RandomState(9012).random()),
        'python_draw': random.Random(1092).random(), 'seeds': [2019,9012,1092]}
