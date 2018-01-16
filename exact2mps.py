import numpy as np
import lib.fci
import obj
import scipy.linalg

def wfn_exact2mps(mol, pbond, config_dic):
    '''
    resort the indirect config dictionary to pbond shape
    [emol1, mol1ph1,mol2ph2,...]
    the mapping relationship is 
    MPS <-> FCI in MPS structure <-> FCI in config_dic structure
    '''
    config_dic_resort = obj.bidict({})
    for key, value in config_dic.iteritems():
        config_new = []
        for imol in xrange(len(mol)):
            offset = len(mol)
            for jmol in xrange(imol):
                offset += mol[jmol].nphs 
            config_new += [value[imol]] + list(value[offset:offset+mol[imol].nphs])
        
        config_dic_resort[key] = tuple(config_new)
    
    mpsfci_configs = lib.fci.fci_configs(None, None, pbond=pbond)
    nmpsfci_configs = len(mpsfci_configs)
    
    mpsfci_exactfci_table = obj.bidict({})
    for idx, iconfig in enumerate(mpsfci_configs):
        if iconfig not in config_dic_resort.inverse:
            mpsfci_exactfci_table[iconfig] = idx+nmpsfci_configs
        else:
            mpsfci_exactfci_table[iconfig] = config_dic_resort.inverse[iconfig]
    
    return mpsfci_exactfci_table


def mpsfci2exactfci(lookuptable, mpsfci, lexactfci):
    
    exactfci = np.zeros([lexactfci],dtype=mpsfci.dtype)
    for i in lookuptable:
        if lookuptable[i] < len(lookuptable):
            exactfci[lookuptable[i]] = mpsfci[i]
    
    #print scipy.linalg.norm(mpsfci), scipy.linalg.norm(exactfci)

    return exactfci


def exactfci2mpsfci(lookuptable, exactfci, pbond):
    
    mpsfci = np.zeros(pbond, dtype=exactfci.dtype)
    for i in lookuptable.inverse:
        if i<len(exactfci):
            mpsfci[lookuptable.inverse[i]] = exactfci[i]
    
    #print scipy.linalg.norm(mpsfci), scipy.linalg.norm(exactfci)

    return  mpsfci
