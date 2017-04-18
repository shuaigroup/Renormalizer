#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# use the graphic method to get the exciton configuration
def exciton_string(no, ne):
    '''
     An Introduction to Configuration Interaction Theory 
     C. David Sherrill
     Figure 2(a)
    '''
    x = np.zeros((no+1, ne+1), dtype=np.int32)
    y = np.zeros((no+1, ne+1), dtype=np.int32)
    x[:,0] = 1
    for ie in xrange(ne+1):
        x[ie, ie] = 1
    for ie in xrange(1,ne+1):
        for io in xrange(ie+1,no+1):
            x[io,ie] += x[io-1, ie-1] + x[io-1, ie]

    for ie in xrange(ne):
        for io in xrange(no+1):
            y[io, ie] = x[io, ie+1]
    return x, y

# exciton configuration to idx
def exconfig2idx(config, y):
    idx = 0
    eidx = 0
    oidx = 0
    for i in config:
        if i != 0:
            idx += y[oidx,eidx]
            eidx += 1
        oidx += 1

    return idx

# idx to exciton configuration
def idx2exconfig(idx, x):
    no = x.shape[0]-1
    ne = x.shape[1]-1
    config = []
    
    eidx = ne
    for io in xrange(no,0,-1):
        if idx >= x[io-1,eidx]:
            idx -= x[io-1,eidx]
            eidx -= 1
            config.append(1)
        else:
            config.append(0)
    return config[::-1]

# idx to phonon configuration
def idx2phconfig(idx, ph_dof_list):
    nphtot = len(ph_dof_list)
    config = []

    remainder = idx
    for i in xrange(1,nphtot):
        shang, remainder = divmod(remainder, ph_dof_list[i])
        config.append(shang)
    config.append(remainder)

    return config

# phonon configuration to idx
def phconfig2idx(config, ph_dof_list):
    nphtot = len(ph_dof_list)
    idx = config[nphtot-1]
    for i in xrange(nphtot-1):
        idx += config[i]*ph_dof_list[i+1]
    return idx

# e-ph configuration to idx
def config2idx(config, ph_dof_list, x, y):

    exidx = exconfig2idx(config[0], y)
    phidx = phconfig2idx(config[1], ph_dof_list)

    idx = exidx * ph_dof_list[0] + phidx

    return idx

# idx to e-ph configuration
def idx2config(idx, ph_dof_list, x, y):
    
    exidx, phidx = divmod(idx, ph_dof_list[0])
    exconfig = idx2exconfig(exidx, x)
    
    phconfig = idx2phconfig(phidx, ph_dof_list)

    return [exconfig, phconfig]


if __name__ == '__main__':

    no = 7
    ne = 5
    x, y = exciton_string(no, ne)
    ph_dof_list = [10000, 1000, 100, 10]
    
    idx = 10100
    config = idx2config(idx, ph_dof_list, x, y)
    print config
    print "idx=", idx
    
    idxcal = config2idx(config, ph_dof_list, x, y)
    print "idxcal", idxcal
