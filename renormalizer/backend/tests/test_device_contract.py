"""Explicit-device contract grid; select backend/device through test environment."""
import os

import numpy as np
import pytest

from renormalizer.backend.context import make_context
from renormalizer.backend.tests.numerical_checks import check_error, check_qr, check_svd, check_eigh, check_solve


@pytest.fixture(params=['float32', 'float64', 'complex64', 'complex128'])
def case(request):
    name = os.environ.get('RENO_TEST_BACKEND')
    if name not in ('jax', 'torch', 'cupy'):
        pytest.skip('requires explicitly selected device backend')
    dtype = np.dtype(request.param)
    real = 'float32' if dtype in (np.dtype('float32'), np.dtype('complex64')) else 'float64'
    ctx = make_context(name, device=os.environ.get('RENO_TEST_DEVICE', 'cpu'), real_dtype=real)
    return ctx, dtype


def host(ctx, x, dtype=None):
    assert ctx.adapter.owns(x), 'result must reside on requested native device'
    if dtype is not None:
        assert ctx.adapter.dtype_of(x) == np.dtype(dtype)
    result = ctx.ops.to_numpy(x)
    if dtype is not None:
        assert result.dtype == np.dtype(dtype)
    return result


def test_device_creation_copy_and_scalar(case):
    ctx, dtype = case
    xh = np.arange(6., dtype=dtype).reshape(2,3)
    x = ctx.ops.from_numpy(xh, copy=True)
    np.testing.assert_array_equal(host(ctx,x,dtype), xh)
    y = ctx.ops.array(x, dtype=dtype, copy=False)
    assert y is x
    z = ctx.ops.array(x, dtype=dtype, copy=True)
    assert z is not x
    assert ctx.adapter.dtype_of(ctx.ops.zeros((2,))) == ctx.real_dtype
    assert ctx.adapter.dtype_of(ctx.ops.array([1j])) == ctx.complex_dtype
    if ctx.device != 'cpu' or ctx.adapter.name == 'jax':
        with pytest.raises(ValueError, match='copy=False'):
            ctx.ops.from_numpy(xh, copy=False)
        with pytest.raises(ValueError, match='copy=False'):
            ctx.ops.to_numpy(x, copy=False)
    scalar = ctx.ops.einsum('ij,ij->', x,x)
    assert scalar.shape == ()
    assert ctx.ops.scalar(scalar) == np.einsum('ij,ij->', xh,xh).item()
    ctx.ops.sync()


def test_device_layout_shape_and_arithmetic(case):
    ctx, dtype = case
    h = np.arange(12.,dtype=dtype).reshape(3,4)[:,::-1]
    if dtype.kind == 'c':
        h = h + 1j*h
    x = ctx.ops.from_numpy(h)
    np.testing.assert_array_equal(host(ctx,ctx.ops.transpose(x),dtype), h.T)
    np.testing.assert_array_equal(host(ctx,ctx.ops.reshape(x,(12,)),dtype), h.reshape(12))
    np.testing.assert_array_equal(host(ctx,ctx.ops.conj(x),dtype), h.conj())
    np.testing.assert_array_equal(host(ctx,ctx.ops.real(x),h.real.dtype), h.real)
    np.testing.assert_array_equal(host(ctx,ctx.ops.imag(x),h.real.dtype), h.imag)
    for operation, reference in [('add',np.add),('subtract',np.subtract),('multiply',np.multiply),('divide',np.divide)]:
        y = ctx.ops.ones((4,),dtype=dtype)*2
        out = getattr(ctx.ops,operation)(x,y)
        check_error(host(ctx,out,dtype), reference(h,np.ones(4,dtype=dtype)*2), atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(host(ctx,ctx.ops.sum(x,axis=0,keepdims=True),dtype),h.sum(axis=0,keepdims=True),atol=2e-5)
    check_error(host(ctx,ctx.ops.norm(x),h.real.dtype),np.asarray(np.linalg.norm(h)),atol=2e-5,rtol=2e-5)
    if dtype.kind != 'c':
        assert ctx.ops.scalar(ctx.ops.max(x)) == h.max()
        assert ctx.ops.scalar(ctx.ops.min(x)) == h.min()
    cast = ctx.ops.astype(x,'complex128')
    assert ctx.adapter.dtype_of(cast) == np.complex128


def test_device_contractions_and_linear_algebra(case):
    ctx,dtype = case
    h = np.array([[1,2],[3,5],[2,7]],dtype=dtype)
    if dtype.kind == 'c':
        h[0,0]+=1j
        h[1,1]-=1j
    x = ctx.ops.from_numpy(h)
    q,r = ctx.ops.qr(x)
    check_qr(h,host(ctx,q,dtype),host(ctx,r,dtype))
    u,s,vh = ctx.ops.svd(x)
    check_svd(h,host(ctx,u,dtype),host(ctx,s,h.real.dtype),host(ctx,vh,dtype))
    ah = h.conj().T@h + np.eye(2,dtype=dtype)
    a = ctx.ops.from_numpy(ah)
    w,v = ctx.ops.eigh(a)
    check_eigh(ah,host(ctx,w,h.real.dtype),host(ctx,v,dtype))
    for bh in [np.array([1,2],dtype=dtype),np.array([[1,2,3],[3,4,5]],dtype=dtype)]:
        result = ctx.ops.solve(a,ctx.ops.from_numpy(bh))
        check_solve(ah,bh,host(ctx,result,dtype))
    check_error(host(ctx,ctx.ops.matmul(ctx.ops.transpose(ctx.ops.conj(x)),x),dtype), h.conj().T@h,atol=2e-5,rtol=2e-5)
    batched_h = np.stack([h,h])
    out = ctx.ops.matmul(ctx.ops.from_numpy(batched_h),ctx.ops.ones((2,3),dtype=dtype))
    check_error(host(ctx,out,dtype),batched_h@np.ones((2,3),dtype=dtype),atol=2e-5,rtol=2e-5)


def test_device_ownership_and_promotion(case):
    ctx,dtype=case
    x=ctx.ops.ones((2,2),dtype=dtype)
    with pytest.raises(TypeError,match='array|ownership|device'):
        ctx.ops.matmul(x,np.eye(2,dtype=dtype))
    a=ctx.ops.ones((1,),dtype='float64')
    b=ctx.ops.ones((1,),dtype='complex64')
    assert ctx.adapter.dtype_of(ctx.ops.add(a,b)) == np.complex128
    with pytest.raises(NotImplementedError):
        ctx.ops.qr(x,mode='complete')
    with pytest.raises(NotImplementedError):
        ctx.ops.svd(x,full_matrices=True)
    bad=ctx.ops.from_numpy(np.array([[np.nan]],dtype=dtype))
    with pytest.raises(ValueError,match='finite'):
        ctx.ops.matmul(bad,bad)


def test_device_updates_and_empty(case):
    ctx,dtype=case
    h=np.array([1,2,3],dtype=dtype)
    x=ctx.ops.from_numpy(h)
    idx=np.array([1,1])
    value=ctx.ops.from_numpy(np.array([2,3],dtype=dtype))
    for name,expected in [('at_add',[1,7,3]),('at_sub',[1,-3,3]),('at_mul',[1,12,3])]:
        np.testing.assert_array_equal(host(ctx,getattr(ctx.ops,name)(x,idx,value),dtype),np.array(expected,dtype=dtype))
    np.testing.assert_array_equal(host(ctx,x,dtype),h)
    np.testing.assert_array_equal(host(ctx,ctx.ops.at_set(x,0,2),dtype),np.array([2,2,3],dtype=dtype))
    with pytest.raises(NotImplementedError,match='duplicate'):
        ctx.ops.at_set(x,idx,value)
    empty=ctx.ops.matmul(ctx.ops.zeros((2,0),dtype=dtype),ctx.ops.zeros((0,3),dtype=dtype))
    np.testing.assert_array_equal(host(ctx,empty,dtype),np.zeros((2,3),dtype=dtype))
    assert ctx.ops.scalar(ctx.ops.sum(ctx.ops.zeros((0,),dtype=dtype))) == 0


@pytest.mark.parametrize('triangle',['L','U'])
def test_device_eigh_uses_only_selected_triangle(case,triangle):
    ctx,dtype=case
    a=np.array([[2,20],[1,4]],dtype=dtype)
    if dtype.kind=='c':
        a[0,0]+=3j
        a[1,1]+=5j
        a[0,1]+=2j
        a[1,0]+=1j
    w,v=ctx.ops.eigh(ctx.ops.from_numpy(a),UPLO=triangle)
    check_eigh(a,host(ctx,w,a.real.dtype),host(ctx,v,dtype),UPLO=triangle)


def test_device_complex_order_and_no_primitive_host_fallback(case,monkeypatch):
    ctx,dtype=case
    h=np.array([1,2,2],dtype=dtype)
    if dtype.kind=='c':
        h+=np.array([8j,3j,4j],dtype=dtype)
    x=ctx.ops.from_numpy(h)
    np.testing.assert_array_equal(host(ctx,ctx.ops.max(x),dtype),np.asarray(h.max()))
    np.testing.assert_array_equal(host(ctx,ctx.ops.min(x),dtype),np.asarray(h.min()))
    original=ctx.adapter.to_numpy
    def forbidden(*args,**kwargs):
        raise AssertionError('primitive invoked host conversion')
    monkeypatch.setattr(ctx.adapter,'to_numpy',forbidden)
    out=ctx.ops.matmul(ctx.ops.ones((2,2),dtype=dtype),ctx.ops.eye(2,dtype=dtype))
    assert ctx.adapter.owns(out)
    monkeypatch.setattr(ctx.adapter,'to_numpy',original)


def test_explicit_cupy_context_rng_does_not_consume_global_stream(case):
    ctx,dtype=case
    if ctx.adapter.name!='cupy':
        pytest.skip('CuPy-specific process RNG isolation')
    import cupy as cp
    cp.random.seed(717)
    expected=cp.asnumpy(cp.random.random(4))
    cp.random.seed(717)
    ctx.adapter.random.normal(size=4)
    actual=cp.asnumpy(cp.random.random(4))
    np.testing.assert_array_equal(actual,expected)


def test_device_empty_decompositions_and_solve(case):
    ctx,dtype=case
    a=ctx.ops.zeros((0,3),dtype=dtype)
    q,r=ctx.ops.qr(a)
    assert q.shape==(0,0) and r.shape==(0,3)
    u,s,vh=ctx.ops.svd(a)
    assert u.shape==(0,0) and s.shape==(0,) and vh.shape==(0,3)
    a=ctx.ops.zeros((0,0),dtype=dtype)
    w,v=ctx.ops.eigh(a)
    assert w.shape==(0,) and v.shape==(0,0)
    result=ctx.ops.solve(a,ctx.ops.zeros((0,2),dtype=dtype))
    assert result.shape==(0,2) and ctx.adapter.owns(result)


def test_device_numpy_scalars_preserve_their_actual_dtype(case):
    ctx,_=case
    for dtype,scalar in [('float32',np.float32(1)),('complex64',np.complex64(1j))]:
        x=ctx.ops.ones((1,),dtype=dtype)
        out=ctx.ops.add(x,scalar)
        assert ctx.adapter.dtype_of(out)==np.dtype(dtype)
    x=ctx.ops.zeros((1,),dtype='float64')
    scalar=np.float64(1+2**-40)
    out=ctx.ops.add(x,scalar)
    np.testing.assert_array_equal(host(ctx,out,'float64'),np.array([scalar]))


def test_device_update_rejects_basic_out_of_bounds(case):
    ctx, _ = case
    x = ctx.ops.ones((2, 3))
    for idx in (99, -3, (0, 3), (slice(None), -4)):
        with pytest.raises(IndexError):
            ctx.ops.at_set(x, idx, 3.)
    ctx.ops.at_set(x, (slice(None, 99), 1), 2.)


def test_device_to_numpy_rejects_invalid_copy(case):
    ctx, _ = case
    with pytest.raises(ValueError):
        ctx.ops.to_numpy(ctx.ops.ones((1,)), copy='bad')
