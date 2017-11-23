# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

'''
useful utilities
'''

from itertools import islice, cycle
import numpy as np
import cPickle as pickle

def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


def autocorr_store(autocorr, istep, freq=10):
    if istep % freq == 0:
        autocorr = np.array(autocorr)
        with open("autocorr"+".npy", 'wb') as f:
            np.save(f,autocorr)


def wfn_store(MPS, istep, filename, freq=100):
    if istep % freq == 0:
        with open(filename, 'wb') as f:
            pickle.dump(MPS,f,-1)
