import numpy as np
import pytest
cp = pytest.importorskip("cupy")
from tools.backend_validation.rawkernel import matmul

@pytest.mark.parametrize('shape',[(7,5,3),(1,0,9),(0,4,3),(9,4,0),(17,3,13)])
def test_reviewed_control(shape):
    m,k,n=shape
    a=(np.arange(m*k,dtype='float64').reshape(m,k)%7-3)/7
    b=(np.arange(k*n,dtype='float64').reshape(k,n)%5-2)/5
    import math
    ref=np.array([math.fsum(float(a[i,j])*float(b[j,l]) for j in range(k)) for i in range(m) for l in range(n)],dtype='float64').reshape(m,n)
    result=matmul(cp.asarray(a),cp.asarray(b))
    assert result.device.id==0
    np.testing.assert_allclose(result.get(),ref,rtol=1e-12,atol=1e-12)

def test_rejects_dtype_layout_and_shape():
    with pytest.raises(TypeError): matmul(cp.ones((2,2),dtype='float32'),cp.ones((2,2)))
    with pytest.raises(ValueError): matmul(cp.ones((3,4))[:,::2],cp.ones((2,3)))
    with pytest.raises(ValueError): matmul(cp.ones((2,3)),cp.ones((2,2)))
