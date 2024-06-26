# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

"""
useful utilities
"""
from typing import List, Union


import numpy as np
import scipy


# from https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


class cached_property():
    """
    A property that is only computed once per instance and then replaces itself
    with an ordinary attribute. Deleting the attribute resets the property.
    """

    def __init__(self, func):
        self.__doc__ = getattr(func, "__doc__")
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self

        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


def calc_vn_entropy(p: Union[np.ndarray, List[float]]) -> float:
    # calculate Von Neumann entropy from density matrix eigenvalues (not singular values!)
    p = np.array(p)
    assert np.allclose(p[p<0], 0)
    p = p / p.sum()
    assert np.allclose(p.sum(), 1)
    p = p[0 < p]
    return - (p* np.log(p)).sum()


def calc_vn_entropy_dm(dm: np.ndarray) -> float:
    # calculate Von Neumann entropy from density matrix

    # reshape dm to square matrix
    dim = np.prod(dm.shape[:dm.ndim // 2])
    dm = dm.reshape((dim, dim))
    w, v = scipy.linalg.eigh(dm)
    return calc_vn_entropy(w)
