# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

'''
the 1 to 1 map between a string like configuration and the index of this
configuration in an array
'''

import numpy as np


def exciton_string(no, ne):
    '''
    a graphic method to get the exciton/electronic configuration

    Ref. An Introduction to Configuration Interaction Theory 
    C. David Sherrill Figure 2(a)
    
    x is the vertex weight, y is the arc weight
    vertex(o,e) is the # of configurations = the sum of vertex(o-1,e) and (o-1,e-1)
    arc(e,o) is the arc link (o,e) and (o+1,e+1) and is vertex weight (o, e+1)

    from configuration to index is the sum of arc according to the
    occupation pattern(0,1,...,19,20)
    from index to configuration is from the specific point(no,ne) to the (0,0)
    point, if(no-1,ne) <= index, then no orbital is occupied.

                        e
        0    1    2    3    4    5
       _______________________________
    0|  1
     |  |  \0  
    1|  1    1 
     |  |  \1|  \0 
    2|  1    2    1 
     |     \2|  \1| \0 
    3|       3    3   1
   o |          \3| \1|  \0 
    4|            6   4    1
     |              \4|  \1|  \0 
    5|                10   5    1   
     |                   \5|  \1|
    6|                     15   6
     |                        \6|
    7|                          21
    '''

    assert no >= ne
    x = np.zeros((no+1, ne+1), dtype=np.int32)
    y = np.zeros((no+1, ne+1), dtype=np.int32)
    x[:no-ne+1,0] = 1
    for ie in xrange(ne+1):
        x[ie, ie] = 1
    for ie in xrange(1,ne+1):
        for io in xrange(ie+1,ie+no-ne+1):
            x[io,ie] += x[io-1, ie-1] + x[io-1, ie]

    for ie in xrange(ne):
        for io in xrange(ie+no-ne+1):
            y[io, ie] = x[io, ie+1]
    return x, y


def exconfig2idx(config, y):
    '''
    exciton/electronic configuration to index
    '''
    assert len(config) == y.shape[0]-1
    assert np.sum(config) == y.shape[1]-1

    idx = 0
    eidx = 0
    oidx = 0
    for i in config:
        if i == 1:
            idx += y[oidx,eidx]
            eidx += 1
        oidx += 1

    return idx


def idx2exconfig(idx, x):
    '''
    index to exciton configuration
    '''
    assert idx < x[-1,-1] and idx >= 0

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


def idx2phconfig(idx, ph_dof_list):
    '''
    index to phonon configuration
    ph_dof_list is the phonon level product[...l3*l2*l1, l2*l1, l1]
    '''
    assert idx < ph_dof_list[0] and idx >= 0

    nphtot = len(ph_dof_list)
    config = []

    remainder = idx
    for i in xrange(1,nphtot):
        shang, remainder = divmod(remainder, ph_dof_list[i])
        config.append(shang)
    config.append(remainder)

    return config


def phconfig2idx(config, ph_dof_list):
    '''
    phonon configuration to index
    '''
    assert len(config) == len(ph_dof_list)

    nphtot = len(ph_dof_list)
    idx = config[nphtot-1]
    for i in xrange(nphtot-1):
        idx += config[i]*ph_dof_list[i+1]

    assert idx < ph_dof_list[0]
    
    return idx


def config2idx(config, direct=None, indirect=None):
    '''
    e-ph configuration to index
    direct: input the 1-1 map configuration and index hash table, key/value are tuple
    indirect: input the ph_dof_list, x, y to calculate on the fly
    config[0] is e config
    config[1] is ph config
    '''

    if indirect != None:
        ph_dof_list, x, y = indirect
        exidx = exconfig2idx(config[0], y)
        phidx = phconfig2idx(config[1], ph_dof_list)

        idx = exidx * ph_dof_list[0] + phidx
    
    elif direct != None:
        nmols, config_dic = direct
        totconfig = tuple(config[0])+tuple(config[1])
        if totconfig not in config_dic.inverse:
            return None
        else:
            idx = config_dic.inverse[totconfig]

    return idx


def idx2config(idx, direct=None, indirect=None):
    '''
    index to e-ph configuration
    '''
    if indirect != None:
        ph_dof_list, x, y = indirect
        exidx, phidx = divmod(idx, ph_dof_list[0])
        exconfig = idx2exconfig(exidx, x)
        phconfig = idx2phconfig(phidx, ph_dof_list)

    elif direct != None:
        nmols, config_dic = direct
        if idx not in config_dic:
            return None
        else:
            config = config_dic[idx]
            exconfig = list(config[0:nmols])
            phconfig = list(config[nmols:])

    return [exconfig, phconfig]

