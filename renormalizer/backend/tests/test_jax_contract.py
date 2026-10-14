import importlib.util
import json
import os
import subprocess
import sys
import weakref

import numpy as np
import pytest

pytestmark=pytest.mark.skipif(importlib.util.find_spec('jax') is None,reason='JAX environment required')


@pytest.mark.parametrize('x64', [None, '0', '1'])
@pytest.mark.parametrize('fp32', [False, True])
def test_public_selection_defaults_to_double_precision(x64, fp32):
    # Each process initializes JAX under the requested policy. Changing its
    # global setting in a running test would not test the public startup path.
    code = '''
import json, os, random
import jax
import numpy as np
import renormalizer as reno
from renormalizer.backend.context import make_context

previous = reno.set_backend('numpy')
proxy = reno.backend
np.random.seed(734)
random.seed(829)
numpy_state = np.random.get_state()
python_state = random.getstate()
before_x64 = bool(jax.config.x64_enabled)
fp32 = 'RENO_FP32' in os.environ
selected = reno.set_backend('jax')
assert reno.get_backend() is selected and proxy.current is selected
real = np.float32 if fp32 else np.float64
complex_dtype = np.complex64 if fp32 else np.complex128
assert selected.real_dtype == real and selected.complex_dtype == complex_dtype
value = np.array([1.25 if fp32 else 1 + 2**-40], dtype=real)
actual = selected.asarray(value)
assert actual.dtype == np.dtype(real)
np.testing.assert_array_equal(selected.numpy(actual), value)
value = value.astype(complex_dtype) * (1 + 1j)
actual = selected.asarray(value)
assert actual.dtype == np.dtype(complex_dtype)
np.testing.assert_array_equal(selected.numpy(actual), value)
after = np.random.get_state()
assert after[0] == numpy_state[0]
np.testing.assert_array_equal(after[1], numpy_state[1])
assert after[2:] == numpy_state[2:]
assert random.getstate() == python_state

# Explicit float32 contexts remain legal even if legacy defaults request f64.
ctx = make_context('jax', device='cpu', real_dtype='float32')
assert ctx.ops.ones((1,)).dtype == np.float32
assert bool(jax.config.x64_enabled) == (before_x64 or not fp32)
print(json.dumps({'x64': bool(jax.config.x64_enabled), 'fp32': fp32}))
'''
    env = os.environ.copy()
    env.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
               XLA_PYTHON_CLIENT_PREALLOCATE='false')
    env.pop('JAX_ENABLE_X64', None)
    if x64 is not None:
        env['JAX_ENABLE_X64'] = x64
    env.pop('RENO_FP32', None)
    if fp32:
        env['RENO_FP32'] = ''  # Legacy option is presence-based.
    result = subprocess.run([sys.executable, '-c', code], env=env, text=True,
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        'x64': x64 == '1' or not fp32, 'fp32': fp32}


def test_default_context_enables_double_but_explicit_float32_does_not():
    code='''
import jax,json,numpy as np
from renormalizer.backend.context import make_context
before=bool(jax.config.x64_enabled)
ctx32=make_context('jax',device='cpu',real_dtype='float32')
assert ctx32.ops.ones((1,)).dtype == np.float32
assert not jax.config.x64_enabled
existing = ctx32.ops.ones((1,))
ctx=make_context('jax',device='cpu')
assert ctx.ops.ones((1,)).dtype == np.float64
assert ctx.ops.array([1j]).dtype == np.complex128
assert existing.dtype == np.float32
assert ctx32.ops.ones((1,)).dtype == np.float32
print(json.dumps([before,bool(jax.config.x64_enabled)]))
'''
    env=os.environ.copy()
    env['JAX_ENABLE_X64']='0'
    env['JAX_PLATFORMS']='cpu'
    env['CUDA_VISIBLE_DEVICES']=''
    env.pop('RENO_FP32', None)
    env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    result=subprocess.run([sys.executable,'-c',code],env=env,text=True,capture_output=True,timeout=60,check=True)
    before,after=json.loads(result.stdout.strip().splitlines()[-1])
    assert before is False and after is True


def test_failed_x64_configuration_does_not_publish_backend():
    code = '''
import jax, numpy as np, random
import renormalizer as reno
from renormalizer.backend.contracts import PrecisionError
previous = reno.set_backend('numpy')
numpy_state = np.random.get_state()
python_state = random.getstate()
# A configuration operation that did not take effect must fail closed.
jax.config.update = lambda *args: None
try:
    reno.set_backend('jax')
except PrecisionError:
    pass
else:
    raise AssertionError('unavailable double precision was accepted')
assert reno.get_backend() is previous and reno.backend.current is previous
after = np.random.get_state()
assert after[0] == numpy_state[0] and after[2:] == numpy_state[2:]
np.testing.assert_array_equal(after[1], numpy_state[1])
assert random.getstate() == python_state
assert not jax.config.x64_enabled
'''
    env = os.environ.copy()
    env.update(JAX_ENABLE_X64='0', JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='')
    env.pop('RENO_FP32', None)
    result = subprocess.run([sys.executable, '-c', code], env=env, text=True,
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_instance_rng_precision_and_sync_does_not_retain_arrays():
    # JAX import itself may consume host RNG state; measure the adapter after
    # that third-party initialization, not package-import side effects.
    import jax
    from renormalizer.backend.context import make_context
    state=np.random.get_state()
    ctx=make_context('jax',device='cpu',real_dtype='float64')
    random=ctx.adapter.random.normal(size=3)
    assert ctx.adapter.owns(random) and random.dtype==np.float64
    values=ctx.adapter.random.randint(5,size=3)
    assert ctx.adapter.owns(values)
    assert bool((values>=0).all()) and bool((values<5).all())
    x=ctx.ops.ones((3,))
    reference=weakref.ref(x)
    ctx.ops.sync()
    del x
    assert reference() is None
    after=np.random.get_state()
    np.testing.assert_array_equal(state[1],after[1])
    assert state[2:]==after[2:]


def test_sync_waits_for_live_random_result(monkeypatch):
    from renormalizer.backend.context import make_context
    ctx=make_context('jax',device='cpu',real_dtype='float64')
    value=ctx.adapter.random.normal(size=3)
    waited=[]
    original=type(value).block_until_ready
    def wait(array):
        waited.append(id(array))
        return original(array)
    monkeypatch.setattr(type(value),'block_until_ready',wait)
    ctx.ops.sync()
    assert id(value) in waited


def test_sync_waits_for_live_legacy_linalg_result(monkeypatch):
    from renormalizer.backend.context import make_context
    ctx=make_context('jax',device='cpu',real_dtype='float64')
    q,r=ctx.adapter.linalg.qr(ctx.ops.ones((3,2)))
    waited=[]
    original=type(q).block_until_ready
    def wait(array):
        waited.append(id(array))
        return original(array)
    monkeypatch.setattr(type(q),'block_until_ready',wait)
    ctx.ops.sync()
    assert id(q) in waited and id(r) in waited


def test_host_upload_does_not_alias_host_storage():
    # Host Matrix storage is mutable; an uploaded array must keep its values even
    # when the host array is updated immediately, before an asynchronous transfer
    # could complete. Repeated with a large buffer so a race shows up reliably.
    from renormalizer.backend.context import make_context
    ctx=make_context('jax',device='cpu',real_dtype='float64')
    expected=np.arange(512*512.).reshape(512,512)
    for copy in (None, True):
        for _ in range(50):
            host=expected.copy()
            uploaded=ctx.adapter.array(host, copy=copy)
            host[...]=-1.
            assert uploaded.dtype==np.float64 and ctx.adapter.owns(uploaded)
            np.testing.assert_array_equal(np.asarray(uploaded), expected)


def test_write_owned_rows_match_functional_update_and_sync_skips_donated():
    from renormalizer.backend.context import make_context
    ctx=make_context('jax',device='cpu',real_dtype='float64')
    adapter=ctx.adapter
    expected=np.zeros((4,3), dtype=complex)
    workspace=adapter.zeros((4,3), dtype=np.complex128)
    for row in (0, np.int64(2), -1):
        value=np.full(3, 1.5+row*1j)
        expected[row]=value
        workspace=adapter.write_owned(workspace, row, value)
    np.testing.assert_array_equal(np.asarray(workspace), expected)
    # Slices keep the functional path.
    workspace=adapter.write_owned(workspace, slice(1, 2), np.ones((1,3)))
    expected[1:2]=1
    np.testing.assert_array_equal(np.asarray(workspace), expected)
    # A still-referenced donated buffer must not break sync.
    donated=workspace
    workspace=adapter.write_owned(donated, 3, np.zeros(3))
    adapter.sync()
    expected[3]=0
    np.testing.assert_array_equal(np.asarray(workspace), expected)


def test_environments_stay_native_only_on_cpu():
    # Device memory is scarce: on GPU, workspaces kept across steps stay on the
    # host and are uploaded per site.
    from types import SimpleNamespace
    from renormalizer.backend.jax_backend import JaxBackend
    adapter = JaxBackend('cpu')
    assert adapter.native_workspace_storage is True
    adapter._device = SimpleNamespace(platform='gpu')
    assert adapter.native_workspace_storage is False
