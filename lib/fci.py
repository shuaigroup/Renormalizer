import numpy as N
import scipy.linalg
from itertools import izip

def fci_configs(nsites,pdim):
    """
    int,int -> tuple(int)
    Generate list of all possible configs
    associated with a given number of sites
    """
    return [tuple(config) for config in 
            N.ndindex(tuple([pdim]*nsites))]

def fci_mps(fci,trunc=1.e-12):
    """
    convert fci->mps
    
    truncates by singular value if trunc<1,
    else, truncate by max m
    
    returns *left* canonicalized MPS
    """
    assert trunc>0

    mps=[]
    pdim=fci.shape[0] # phys dim
    nsites=len(fci.shape)

    residual=N.reshape(fci,[pdim,N.prod(fci.shape[1:])])
    
    for i in xrange(nsites-1):
        u,sigma,vt=scipy.linalg.svd(residual,full_matrices=False)
        if trunc<1.:
            # count how many sing vals < trunc            
            m_trunc=len([s for s in sigma if s >trunc])
        else:
            m_trunc=int(trunc)
            m_trunc=min(m_trunc,len(sigma))

        u=u[:,0:m_trunc]
        sigma=N.diag(sigma[0:m_trunc])
        vt=vt[0:m_trunc,:]

        residual=N.dot(sigma,vt)
        residual=N.reshape(residual,[m_trunc*pdim,
                                     vt.shape[1]/pdim])
        
        mpsi=N.reshape(u,[u.shape[0]/pdim,pdim,m_trunc])
        mps.append(mpsi)

    # last site, append residual
    mpsi=N.reshape(residual,[m_trunc,pdim,1])
    mps.append(mpsi)
    return mps
