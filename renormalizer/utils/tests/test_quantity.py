# -*- coding: utf-8 -*-

from pytest import approx
from renormalizer.utils import Quantity


def test_quantity():
    q1 = Quantity(1, "a.u.")
    q2 = q1.as_unit("cm-1")
    assert approx(q2.value, rel=1e-4) == 2.1947e5
    assert approx(q2.as_au(), rel=1e-4) == 1