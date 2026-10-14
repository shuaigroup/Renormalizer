"""Additional consumed signatures discovered by frozen domain regression."""
import numpy as np
import pytest


def host(ctx,value):
    assert isinstance(value,(np.ndarray,np.generic)) if ctx.adapter.name=='numpy' else ctx.adapter.owns(value)
    return np.asarray(ctx.adapter.numpy(value))


@pytest.mark.parametrize('value',[-.2,-.2j,np.array([-.2, .3]),np.array([-.2j,.3+0.4j])])
def test_absolute_consumed_scalar_and_vector(captured_backend,value):
    ctx=captured_backend
    arg=ctx.ops.from_numpy(value) if isinstance(value,np.ndarray) else value
    np.testing.assert_allclose(host(ctx,ctx.adapter.absolute(arg)),np.absolute(value))


@pytest.mark.parametrize('a,b',[(-.1,.2),(-.1j,.2j)])
def test_tdvp_min_abs_time_step(a,b):
    from renormalizer.mps.mps import min_abs
    assert min_abs(a,b)==a
    assert min_abs(b,a)==a


@pytest.mark.parametrize('axes',[(range(2,4),range(2,4)),([2,3],[2,3]),2,np.int64(2)])
def test_tensordot_consumed_iterables(captured_backend,axes):
    ctx=captured_backend
    a=np.arange(16.).reshape(2,2,2,2)
    b=np.arange(16.).reshape(2,2,2,2)
    out=ctx.adapter.tensordot(ctx.ops.from_numpy(a),ctx.ops.from_numpy(b),axes=axes)
    np.testing.assert_allclose(host(ctx,out),np.tensordot(a,b,axes=axes))


@pytest.mark.parametrize('operation',['argmax','repeat','nonzero','equal','unique'])
def test_num_jac_consumed_helpers(captured_backend,operation):
    ctx=captured_backend;b=ctx.adapter
    h=np.array([[1.,0.,3.],[0.,2.,0.]])
    x=ctx.ops.from_numpy(h)
    if operation=='argmax':
        actual,expected=b.argmax(x,axis=0),np.argmax(h,axis=0)
    elif operation=='repeat':
        actual,expected=b.repeat(ctx.ops.array([0,1]),3),np.repeat([0,1],3)
    elif operation=='nonzero':
        actual,expected=b.nonzero(x),np.nonzero(h)
        assert isinstance(actual,tuple) and len(actual)==2
        for left,right in zip(actual,expected):np.testing.assert_array_equal(host(ctx,left),right)
        return
    elif operation=='equal':
        actual,expected=b.equal(x,0),np.equal(h,0)
    else:
        actual,expected=b.unique(ctx.ops.array([2,0,2,1])),np.unique([2,0,2,1])
    np.testing.assert_array_equal(host(ctx,actual),expected)


def test_internal_dense_num_jac_analytic(captured_backend):
    from renormalizer.lib.integrate._ivp.common import num_jac
    ctx=captured_backend
    matrix=ctx.ops.array([[-1.,0.],[0.,-2.]])
    y=ctx.ops.array([1.,2.])
    fun=lambda t,x:matrix@x
    jac,factor=num_jac(fun,0.,y,fun(0.,y),1e-9,None)
    np.testing.assert_allclose(host(ctx,jac),[[-1.,0.],[0.,-2.]],rtol=1e-6,atol=1e-8)


def test_full_consumed_scalar_shape(captured_backend):
    ctx=captured_backend
    out=ctx.adapter.full(3,np.float64(.25))
    np.testing.assert_array_equal(host(ctx,out),[.25,.25,.25])


@pytest.mark.parametrize('reuse_factor',[False,True])
def test_internal_sparse_num_jac_analytic(captured_backend,reuse_factor):
    from scipy.sparse import csc_matrix,isspmatrix_csc
    from renormalizer.lib.integrate._ivp.common import num_jac
    ctx=captured_backend
    expected=np.array([[-1.,2.,0.],[0.,-2.,0.],[0.,0.,-3.]])
    matrix=ctx.ops.from_numpy(expected);y=ctx.ops.array([1.,2.,3.])
    fun=lambda t,x:matrix@x
    original=np.array([1e-12]*3)
    factor=ctx.ops.from_numpy(original) if reuse_factor else None
    jac,out_factor=num_jac(fun,0.,y,fun(0.,y),1e-9,factor,
                           (csc_matrix(expected!=0),np.array([0,1,0])))
    assert isspmatrix_csc(jac),'sparse assembly has an explicit SciPy host return contract'
    np.testing.assert_allclose(jac.toarray(),expected,rtol=1e-3,atol=1e-4)
    assert host(ctx,out_factor).shape==(3,)
    if reuse_factor:np.testing.assert_array_equal(host(ctx,factor),original)


@pytest.mark.parametrize('sparse',[False,True])
def test_num_jac_repairs_underflow_step_without_mutating_factor(captured_backend,sparse):
    from scipy.sparse import csc_matrix
    from renormalizer.lib.integrate._ivp.common import num_jac
    ctx=captured_backend
    expected=np.diag([-1.,-2.,-4.]);matrix=ctx.ops.from_numpy(expected)
    y=ctx.ops.array([1.,2.,4.]);factor=ctx.ops.array([1e-20]*3)
    def fun(t,x):
        return matrix@x
    sparsity=(csc_matrix(expected!=0),np.zeros(3,dtype=int)) if sparse else None
    jac,new_factor=num_jac(fun,0.,y,fun(0.,y),1e-9,factor,sparsity)
    np.testing.assert_allclose(jac.toarray() if sparse else host(ctx,jac),expected,rtol=1e-8,atol=1e-10)
    np.testing.assert_array_equal(host(ctx,factor),[1e-20]*3)
    # The repaired step is visible in the returned factor; the caller's is untouched.
    assert np.all(host(ctx,new_factor)>1e-20)
