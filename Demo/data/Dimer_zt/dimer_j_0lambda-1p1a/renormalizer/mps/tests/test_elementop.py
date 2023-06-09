# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import pytest

import numpy as np

from renormalizer.utils import elementop

test_phonon_op_data = (
    [r"b^\dagger b", 3, 3, 3.0],
    [r"b^\dagger b", 3, 2, 0.0],
    [r"b^\dagger b", 2, 3, 0.0],
    [r"b^\dagger b", 0, 0, 0.0],
    [r"b^\dagger + b", 3, 3, 0.0],
    [r"b^\dagger + b", 3, 2, np.sqrt(3.0)],
    [r"b^\dagger + b", 2, 3, np.sqrt(3.0)],
    [r"b^\dagger + b", 4, 2, 0.0],
    [r"b^\dagger + b", 2, 4, 0.0],
    ["Iden", 2, 4, 0.0],
    ["Iden", 4, 2, 0.0],
    ["Iden", 2, 2, 1.0],
)


@pytest.mark.parametrize("op, bra, ket, value", test_phonon_op_data)
def test_phonon_op(op, bra, ket, value):
    assert elementop.ph_element_op(op, bra, ket) == pytest.approx(value)


def test_phonon_exception():
    with pytest.raises(AssertionError):
        elementop.ph_element_op(r"b^\dagger b", 0, -1)


test_electronic_op_data = (
    [r"a^\dagger", 1, 0, 1.0],
    [r"a^\dagger", 0, 1, 0.0],
    [r"a^\dagger", 1, 1, 0.0],
    [r"a^\dagger", 0, 0, 0.0],
    ["a", 1, 0, 0.0],
    ["a", 0, 1, 1.0],
    ["a", 1, 1, 0.0],
    ["a", 0, 0, 0.0],
    [r"a^\dagger a", 1, 0, 0.0],
    [r"a^\dagger a", 0, 1, 0.0],
    [r"a^\dagger a", 1, 1, 1.0],
    [r"a^\dagger a", 0, 0, 0.0],
    ["Iden", 1, 0, 0.0],
    ["Iden", 0, 1, 0.0],
    ["Iden", 1, 1, 1.0],
    ["Iden", 0, 0, 1.0],
)


@pytest.mark.parametrize("op, bra, ket, value", test_electronic_op_data)
def test_electronic_op(op, bra, ket, value):
    assert elementop.e_element_op(op, bra, ket) == pytest.approx(value)


def test_electronic_exception():
    with pytest.raises(AssertionError):
        elementop.e_element_op("a", 0, 3)
