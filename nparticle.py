# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
from configidx import *
import itertools
import obj

'''
based on the n-particle approximation
for 1 exciton system, only (n-1) GS states allow phonon excitation
'''

def construct_config_dict(mol, nexciton,  nparticle=False):
    '''
    construct the configuration dictionary (1-1 map between string like configuration and
    index in a vector, two way hash table)

    The structure of the configuration string is a tuple (1,0,1...,7,8,9...)
    the first nmols terms are electronic config: mol1,mol2,mol3...
    term is the phonon config mol1ph1, mol1ph2,...mol2ph1....
    '''
    
    nmols = len(mol)

    if nparticle == False:
        ngs_vib_allow = nmols - nexciton
    else:
        assert nparticle <= nmols and nparticle >= nexciton
        ngs_vib_allow = nparticle - nexciton
    
    config_dic = obj.bidict({})
    config_dic_key = -1
    
    for exlist in itertools.combinations(range(nmols), nexciton):
        # construct the ex config
        exiconfig = [1 if x in exlist else 0 for x in xrange(nmols)]

        ex_vib_allow = [i for i,j in enumerate(exiconfig) if j != 0]
        gs_vib = [i for i,j in enumerate(exiconfig) if j == 0]
    
        for gs_vib_allow in itertools.combinations(gs_vib, ngs_vib_allow):
            vib_allow = sorted(ex_vib_allow + list(gs_vib_allow))
            
            phlist = []
            for imol in xrange(nmols):
                for iph in xrange(mol[imol].nphs):
                    if imol in vib_allow:
                        phlist.append(range(mol[imol].ph[iph].nlevels)) 
                    else:
                        phlist.append([0]) 
            for phiconfig in itertools.product(*phlist):
                # construct the ph config
                newconfig = tuple(exiconfig) + phiconfig
                if newconfig not in config_dic.inverse:
                    # there have some redundant terms
                    # for example: 3 mol 1phonon each mol 
                    # (1,0,0,1,0,0) may occure when 1st and 2nd mol is excited
                    # or 1st and 3rd mol is excited
                    config_dic_key += 1
                    config_dic[config_dic_key] = newconfig
    
    # check if the string and index is 1 to 1 map in the bi-dictionary
    assert len(config_dic) == len(config_dic.inverse)
    
    return config_dic
    

