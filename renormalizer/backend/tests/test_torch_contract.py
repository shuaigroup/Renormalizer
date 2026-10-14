import importlib.util

import numpy as np
import pytest

pytestmark=pytest.mark.skipif(importlib.util.find_spec('torch') is None,reason='Torch environment required')


def test_legacy_moveaxis_is_native_and_preserves_values():
    from renormalizer.backend.context import make_context
    ctx=make_context('torch',device='cpu',real_dtype='float64')
    h=np.arange(24.).reshape(2,3,4)
    x=ctx.ops.from_numpy(h)
    out=ctx.adapter.moveaxis(x,0,2)
    assert ctx.adapter.owns(out)
    np.testing.assert_array_equal(ctx.ops.to_numpy(out),np.moveaxis(h,0,2))


def test_cpu_copy_promises_and_conjugate_materialization():
    from renormalizer.backend.context import make_context
    ctx=make_context('torch',device='cpu',real_dtype='float64')
    h=np.arange(6.)
    x=ctx.ops.from_numpy(h,copy=False)
    assert np.shares_memory(ctx.ops.to_numpy(x,copy=False),h)
    with pytest.raises(ValueError,match='copy=False'):
        ctx.ops.from_numpy(h[::-1],copy=False)
    h.flags.writeable=False
    with pytest.raises(ValueError,match='copy=False'):
        ctx.ops.from_numpy(h,copy=False)
    c=ctx.ops.conj(ctx.ops.array([1+2j]))
    with pytest.raises(ValueError,match='copy=False'):
        ctx.ops.to_numpy(c,copy=False)
    np.testing.assert_array_equal(ctx.ops.to_numpy(c),[1-2j])


def test_context_does_not_change_torch_defaults_or_rng():
    import torch
    from renormalizer.backend.context import make_context
    state=torch.random.get_rng_state().clone()
    dtype=torch.get_default_dtype()
    ctx=make_context('torch',device='cpu',real_dtype='float64')
    out=ctx.adapter.random.normal(size=3)
    assert ctx.adapter.dtype_of(out)==np.float64
    assert torch.get_default_dtype()==dtype
    assert torch.equal(state,torch.random.get_rng_state())


def test_legacy_vdot_flattens_and_conjugates_first_operand():
    from renormalizer.backend.context import make_context
    ctx=make_context('torch',device='cpu',real_dtype='float64')
    a=np.array([[1+2j,3-1j]])
    b=np.array([[4-1j,2+3j]])
    out=ctx.adapter.vdot(ctx.ops.from_numpy(a),ctx.ops.from_numpy(b))
    assert ctx.adapter.owns(out)
    assert ctx.ops.scalar(out)==np.vdot(a,b)


def test_legacy_elementwise_extrema_promote_inputs():
    from renormalizer.backend.context import make_context
    ctx=make_context('torch',device='cpu',real_dtype='float64')
    a=ctx.ops.array([1,4],dtype='float32')
    b=ctx.ops.array([3,2],dtype='float64')
    maximum=ctx.adapter.maximum(a,b)
    minimum=ctx.adapter.minimum(a,b)
    assert ctx.adapter.dtype_of(maximum)==np.float64
    np.testing.assert_array_equal(ctx.ops.to_numpy(maximum),[3,4])
    np.testing.assert_array_equal(ctx.ops.to_numpy(minimum),[1,2])


@pytest.mark.parametrize('preset', [None, 'ACTIVE'])
def test_openmp_wait_policy_defaults_to_passive_before_torch_loads(preset):
    # Spinning Torch/MKL OpenMP workers starve interleaved OpenBLAS threads;
    # the default only applies before Torch loads and never overrides a choice.
    import os, subprocess, sys
    code = '''
import os, sys
assert 'torch' not in sys.modules
from renormalizer.backend.torch_backend import TorchBackend
TorchBackend()
print(os.environ.get('OMP_WAIT_POLICY'))
'''
    env = {k: v for k, v in os.environ.items() if k != 'OMP_WAIT_POLICY'}
    if preset is not None:
        env['OMP_WAIT_POLICY'] = preset
    result = subprocess.run([sys.executable, '-c', code], env=env, text=True,
                            capture_output=True, timeout=120, check=True)
    assert result.stdout.strip().splitlines()[-1] == (preset or 'PASSIVE')
