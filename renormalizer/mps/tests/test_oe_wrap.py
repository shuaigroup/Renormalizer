"""Error-reporting unit tests deliberately do not exhaust host/device memory.

Native allocator failure behavior belongs in an opt-in process-isolated runtime
probe. These cases test the wrapper's contract, including exception identity and
avoiding misclassification of ordinary runtime failures.
"""
from unittest.mock import Mock

import numpy as np
import pytest

from renormalizer.mps import oe_contract_wrap as wrapper


@pytest.mark.parametrize('expression', [False, True])
def test_memory_error_is_logged_and_reraised(monkeypatch, caplog, expression):
    error = MemoryError('controlled allocation failure')
    operation = Mock(side_effect=error)
    operand = np.ones((2, 2))
    if expression:
        monkeypatch.setattr(wrapper.oe, 'contract_expression', Mock(return_value=operation))
        call = wrapper.oe_contract_expression('ij,jk->ik', operand.shape, operand.shape)
        args = (operand, operand)
        expected = 'Out of memory error calling oe contract expression'
    else:
        monkeypatch.setattr(wrapper.oe, 'contract', operation)
        call = wrapper.oe_contract
        args = ('ij,jk->ik', operand, operand)
        expected = 'Out of memory error calling oe.contract'
    with pytest.raises(MemoryError) as caught:
        call(*args)
    assert caught.value is error
    assert operation.call_count == 1
    assert expected in caplog.text
    assert 'The arguments are:' in caplog.text


@pytest.mark.parametrize('expression', [False, True])
def test_non_memory_error_is_not_reclassified(monkeypatch, caplog, expression):
    error = RuntimeError('invalid contraction')
    operation = Mock(side_effect=error)
    operand = np.ones((2, 2))
    if expression:
        monkeypatch.setattr(wrapper.oe, 'contract_expression', Mock(return_value=operation))
        call = wrapper.oe_contract_expression('ij,jk->ik', operand.shape, operand.shape)
        args = (operand, operand)
    else:
        monkeypatch.setattr(wrapper.oe, 'contract', operation)
        call = wrapper.oe_contract
        args = ('ij,jk->ik', operand, operand)
    with pytest.raises(RuntimeError) as caught:
        call(*args)
    assert caught.value is error
    assert 'Out of memory' not in caplog.text
