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


def autocorr_store(autocorr, istep, index="", freq=10):
    if istep % freq == 0:
        autocorr = np.array(autocorr)
        with open("autocorr"+str(index)+".npy", 'wb') as f:
            np.save(f,autocorr)


def wfn_store(MPS, istep, filename, freq=100):
    if istep % freq == 0:
        with open(filename, 'wb') as f:
            pickle.dump(MPS,f,-1)
            
            
def RK_IVP(v0, func, dt, tableau):
    # Runge-Kutta solver
    # func is dy/dt = func(y)

    klist=[v0,]
    for istage in range(len(tableau[0])):
        v = copy.deepcopy(v0)
        for iv in range(1,len(klist)):
            v += klist[iv]*tableau[0][istage][iv-1]*dt
        
        klist.append(func(v))
    
    v = copy.deepcopy(v0)
    for iv in range(1,len(klist)):
        v += klist[iv]*tableau[1][iv-1]*dt

    return v
