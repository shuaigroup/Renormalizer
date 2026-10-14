from tools.backend_validation.policy import ReviewRecord, ExecutionPolicy, admit, review_source
from tools.backend_validation.sandbox import probe, launch
import pytest

def test_untrusted_is_fail_closed_even_with_boolean_claim(tmp_path):
    path=tmp_path/'candidate.py'; path.write_text('def candidate(a,b): return a @ b\n')
    policy=ExecutionPolicy()
    for device in ['cpu','cuda:0']:
        decision=admit('untrusted',device,candidate=path,policy=policy)
        assert not decision.allowed
    with pytest.raises(TypeError):
        ExecutionPolicy(isolation_verified=True)
    assert not probe().verified
    with pytest.raises((ValueError, KeyError)): launch({})

def test_review_pins_source_and_dependencies(tmp_path):
    source=tmp_path/'candidate.py'; source.write_text('def candidate(a,b): return a @ b\n')
    dep=tmp_path/'helper.py'; dep.write_text('VALUE=1\n')
    review=review_source(source, dependencies=[dep])
    policy=ExecutionPolicy(review=review)
    assert admit('reviewed','cpu',candidate=source,policy=policy).allowed
    dep.write_text('VALUE=2\n')
    assert not admit('reviewed','cpu',candidate=source,policy=policy).allowed
    source.write_text('def candidate(a,b): return a * b\n')
    assert not admit('reviewed','cpu',candidate=source,policy=policy).allowed
