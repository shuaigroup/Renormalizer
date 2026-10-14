import numpy as np
import pytest
from tools.backend_validation.scoring import score_array

TOL = dict(atol_F=1e-12, rtol_F=1e-10, atol_max=1e-12, rtol_max=1e-10)

@pytest.mark.parametrize('a,b,reason', [
    (np.ones(2), np.zeros(2), 'numerical_error'),
    (np.ones(2, dtype='float32'), np.ones(2), 'dtype_mismatch'),
    (np.ones((1,2)), np.ones(2), 'shape_mismatch'),
    (np.array([np.nan]), np.ones(1), 'nonfinite'),
    (np.array([np.inf]), np.ones(1), 'nonfinite'),
])
def test_reject(a, b, reason):
    result = score_array(a, b, tolerances=TOL)
    assert result['status'] == 'fail' and result['reason'] == reason

def test_exact_empty_and_zero_reference():
    for value in [np.array(0.), np.zeros((0,3)), np.array([1+2j])]:
        assert score_array(value, value, tolerances=TOL)['status'] == 'pass'
    result = score_array(np.array([1e-13]), np.zeros(1), tolerances=TOL)
    assert result['status'] == 'pass'
    assert result['metrics']['relative_frobenius'] == 'infinity'

def test_maximum_gate_independent_and_overflow_fail_closed():
    reference = np.ones(100)
    candidate = reference.copy(); candidate[0] += .1
    tol = dict(atol_F=1.,rtol_F=0.,atol_max=.01,rtol_max=0.)
    assert score_array(candidate, reference, tolerances=tol)['status'] == 'fail'
    tol = dict(atol_F=1e308,rtol_F=1e308,atol_max=1e308,rtol_max=1e308)
    assert score_array(np.ones(1), np.ones(1), tolerances=tol)['reason'] == 'verification_overflow'

def test_stable_norm_and_invalid_tolerance():
    a = np.array([1e200, 1e200])
    assert score_array(a, a, tolerances=TOL)['status'] == 'pass'
    with pytest.raises(ValueError):
        score_array(a, a, tolerances={**TOL,'atol_F': float('inf')})


@pytest.mark.parametrize('dtype',['int64','float16'])
def test_shared_arithmetic_does_not_expand_candidate_domain(dtype):
    value=np.ones(1,dtype=dtype)
    assert score_array(value,value,tolerances=TOL)['reason']=='unsupported_dtype'


def test_candidate_tolerances_remain_frozen_builtin_numbers():
    with pytest.raises(ValueError):
        score_array(np.ones(1),np.ones(1),tolerances={**TOL,'atol_F':np.float64(1e-12)})


@pytest.mark.parametrize('key', list(TOL))
@pytest.mark.parametrize('value', [10**400, -(10**400)], ids=['positive_overflow', 'negative_overflow'])
def test_out_of_range_integer_tolerances_rejected(key, value):
    with pytest.raises(ValueError, match='finite nonnegative numbers'):
        score_array(np.ones(1), np.ones(1), tolerances={**TOL, key: value})
