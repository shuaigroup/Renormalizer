"""CPU-only control for selected-device synchronization; not GPU evidence."""
import importlib
from types import SimpleNamespace


def test_sync_uses_device_captured_at_construction(monkeypatch):
    module = importlib.import_module('renormalizer.backend.cupy_backend')
    synchronized = []
    active = [3]
    class Device:
        def __init__(self, device_id=None):
            self.id = active[0] if device_id is None else device_id
        def synchronize(self):
            synchronized.append(self.id)
    fake = SimpleNamespace(ndarray=type('FakeArray', (), {}), linalg=object(), random=object(),
            cuda=SimpleNamespace(Device=Device,
                memory=SimpleNamespace(OutOfMemoryError=MemoryError)))
    monkeypatch.setattr(module, '_cupy', fake)
    monkeypatch.setattr(module, '_cupy_available', True)
    adapter = module.CupyBackend()
    active[0] = 5
    adapter.sync()
    assert synchronized == [3]


def _fake_cupy(current):
    """Fake CuPy whose Device context makes its id current, like cupy.cuda.Device."""
    class Device:
        def __init__(self, device_id=None):
            self.id = current[0] if device_id is None else device_id
        def __enter__(self):
            self._previous = current[0]
            current[0] = self.id
            return self
        def __exit__(self, *exc):
            current[0] = self._previous
        def synchronize(self):
            pass
    record = lambda *args, **kwargs: current[0]
    return SimpleNamespace(
        ndarray=type('FakeArray', (), {}),
        linalg=SimpleNamespace(norm=record, LinAlgError=type('LinAlgError', (Exception,), {})),
        random=SimpleNamespace(seed=record),
        cuda=SimpleNamespace(Device=Device, runtime=SimpleNamespace(getDeviceCount=lambda: 4),
                             memory=SimpleNamespace(OutOfMemoryError=MemoryError)))


def test_bound_algorithm_and_namespaces_run_on_selected_device(monkeypatch):
    # CuPy launches every kernel on the current device; an adapter for another
    # device must make its own device current, not only for its own calls.
    from renormalizer.backend.execution import bind_backend
    from renormalizer.backend.context import capture_backend
    module = importlib.import_module('renormalizer.backend.cupy_backend')
    current = [0]
    monkeypatch.setattr(module, '_cupy', _fake_cupy(current))
    monkeypatch.setattr(module, '_cupy_available', True)
    adapter = module.CupyBackend(device='cuda:2')

    assert adapter.linalg.norm() == 2 and adapter.random.seed(1) == 2
    assert issubclass(adapter.linalg.LinAlgError, Exception)  # non-callables pass through
    assert current == [0]

    @bind_backend
    def algorithm():
        return current[0]
    with capture_backend(adapter):
        assert algorithm() == 2
    assert current == [0]
