from unittest.mock import patch
import pytest

from renormalizer.mps.oe_contract_wrap import oe_contract, oe_contract_expression
from renormalizer.mps.backend import np, MEMORY_ERRORS


def test_oe_contract():
    with patch("logging.Logger.fatal") as mock_logger_fatal:
        with pytest.raises(MEMORY_ERRORS):
            a = np.random.rand(2<<20)
            oe_args = []
            for i in range(5):
                oe_args.extend([a, [i]])
            oe_args.append(list(range(5)))
            oe_contract(*oe_args)

        # Verify that logger.fatal was called multiple times
        assert mock_logger_fatal.call_count > 1, "logger.fatal was not called multiple times"
        # Verify that one of the calls contains the specific message
        messages = [call[0][0] for call in mock_logger_fatal.call_args_list]
        assert "Out of memory error calling oe.contract" in messages, (
            "Expected message not found in logger.fatal calls"
        )


def test_oe_contract_expression():
    with patch("logging.Logger.fatal") as mock_logger_fatal:
        with pytest.raises(MEMORY_ERRORS):
            a = np.random.rand(2 << 20)
            expr = oe_contract_expression(
                "a, b, c, d, e -> abcde",
                a, a, a, a, (2 << 20, ),
                constants=[0, 1, 2, 3])
            expr(a)

        # Verify that logger.fatal was called multiple times
        assert mock_logger_fatal.call_count > 1, "logger.fatal was not called multiple times"
        # Verify that one of the calls contains the specific message
        messages = [call[0][0] for call in mock_logger_fatal.call_args_list]
        assert "Out of memory error calling oe contract expression" in messages, (
            "Expected message not found in logger.fatal calls"
        )

