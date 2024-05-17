# -*- coding: utf-8 -*-
# Author: Yu Xiong <y-xiong22@mails.tsinghua.edu.cn>

import pytest

from renormalizer.mps import backend


def set_tolerance(tolerance_type, value):
    setattr(backend, tolerance_type, value)


def get_tolerance(tolerance_type):
    return getattr(backend, tolerance_type)


@pytest.mark.parametrize(
    "tolerance_type, value",
    [
        ("canonical_atol", 1e-5),       # normal
        ("canonical_atol", -1e-7),      # ValueError
        ("canonical_atol", "invalid"),  # ValueError
        ("canonical_rtol", 1e-4),       # normal
        ("canonical_rtol", -1e-6),      # ValueError
        ("canonical_rtol", "invalid"),  # ValueError
    ],
)
def test_tolerances(tolerance_type, value):
    if isinstance(value, (int, float)) and value >= 0:
        set_tolerance(tolerance_type, value)
        assert get_tolerance(tolerance_type) == value
    else:
        with pytest.raises(ValueError):
            set_tolerance(tolerance_type, value)
