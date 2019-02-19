# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

"""
useful utilities
"""

from itertools import islice, cycle
import sys

import numpy as np

from ephMPS.utils import pickle


def roundrobin(*iterables):
    """
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    """
    pending = len(iterables)
    if sys.version_info[0] == 3:
        nexts = cycle(iter(it).__next__ for it in iterables)
    else:
        nexts = cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next_func in nexts:
                yield next_func()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


def autocorr_store(autocorr, istep, freq=10):
    if istep % freq == 0:
        autocorr = np.array(autocorr)
        with open("autocorr" + ".npy", "wb") as f:
            np.save(f, autocorr)


def wfn_store(mps, istep, filename, freq=100):
    if istep % freq == 0:
        with open(filename, "wb") as f:
            pickle.dump(mps, f, -1)


# from https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)
