from renormalizer.mps import Mps, Mpo, gs, TDA
from renormalizer.model import Op, Model
from renormalizer.model import basis as ba
from renormalizer.utils.constant import *
from renormalizer.mps.tests import cur_dir

import os
import numpy as np
import itertools
from collections import Counter, defaultdict
import scipy
import logging

logger = logging.getLogger(__name__)

def test_tda():
    from renormalizer.tests.c2h4_para import ff, omega_std, B, zeta
    # See  J. Chem. Phys. 153, 084118 (2020) for the details of the Hamiltonian
    # the order is according to the harmonic frequency from small to large.
    ham_terms = []
    
    nmode = 12
    omega = {}
    
    nterms = 0
    # potential terms
    for term in ff:
        mode, factor = term[:-1], term[-1]
        # ignore the factor smaller than 1e-15
        if abs(factor) < 1e-15:
            continue
        # ignore the expansion larger than 4-th order 
        #if len(mode) > 4:
        #    continue

        mode = Counter(mode)
        
        # the permutation symmetry prefactor
        prefactor = 1.
        for p in mode.values():
            prefactor *= scipy.special.factorial(p, exact=True)
        
        # check harmonic term 
        if len(mode) == 1 and list(mode.values())[0] == 2:
            omega[list(mode.keys())[0]] = np.sqrt(factor)
        
        dof = [f"v_{i}" for i in mode.keys()]
        symbol = " ".join([f"x^{i}" for i in mode.values()])
        qn = [0 for i in mode.keys()]
        factor /= prefactor
        ham_terms.append(Op(symbol, dof, factor=factor, qn=qn))
        nterms += 1
    
    # Coriolis terms
    B = np.array(B)
    zeta = np.array(zeta)
    
    terms = [("x","partialx","x","partialx",1.),
     ("x","partialx","partialx","x",-1.),
     ("partialx","x","x","partialx",-1.),
     ("partialx","x","partialx","x",1.)]
    for j,l in itertools.product(range(nmode),repeat=2):
        for i,k in itertools.product(range(j), range(l)):
            dof = [f"v_{i}", f"v_{j}", f"v_{k}", f"v_{l}"]
            tmp = -np.einsum("i,i,i ->", B, zeta[:,i,j], zeta[:,k,l])
            qn = [0,0,0,0]
            if abs(tmp) < 1e-15:
                continue
            for term in terms:  
                symbol, factor = " ".join(term[:-1]), term[-1]*tmp
                ham_terms.append(Op(symbol, dof, factor=factor, qn=qn))
            nterms += 4
    
    # Kinetic terms
    for imode in range(nmode):
        ham_terms.append(Op("p^2", f"v_{imode}", 0.5, 0))
        nterms += 1

    logger.info(f"nterms: {nterms}")
    logger.info(f"omega: {np.sort(np.array(list(omega.values())),axis=None)*au2cm}")
    logger.info(f"omega_std: {np.array(omega_std)}")
    
    basis = []
    for imode in range(nmode):
        basis.append(ba.BasisSHO(f"v_{imode}", omega[imode], 4, dvr=False))
    
    model = Model(basis, ham_terms)
    mpo = Mpo(model)
    logger.info(f"mpo_bond_dims:{mpo.bond_dims}")
    #assert mpo.is_hermitian()
    
    alias = ["v10","v8","v7","v4","v6","v3","v12","v2","v11","v1","v5","v9"]
    energy_list = {}
    M=10
    procedure = [[M, 0.4], [M, 0.2], [M, 0.2], [M, 0.1]] + [[M, 0]]*100
    mps = Mps.random(model, 0, M, percent=1.0)
    mps.optimize_config.procedure = procedure
    mps.optimize_config.method = "2site"
    mps.optimize_config.e_rtol = 1e-6
    mps.optimize_config.e_atol = 1e-8
    mps.optimize_config.nroots = 1
    energies, mps = gs.optimize_mps(mps, mpo)
    logger.info(f"M: {M}, energy : {np.array(energies[-1])*au2cm}")
    tda = TDA(model, mpo, mps, nroots=3)
    e = tda.kernel(include_psi0=False) 
    logger.info(f"tda energy : {(e-energies[-1])*au2cm}")
    assert np.allclose((e-energies[-1])*au2cm, [824.74925026, 936.42650242, 951.96826289], atol=1)
    config, compressed_mps = tda.analysis_dominant_config(alias=alias)
    # std is calculated with M=200, include_psi0=True; the initial gs is
    # calculated with 9 state SA-DMRG; physical_bond=6 
    std = np.load(os.path.join(cur_dir, "c2h4_std.npz"))["200"]
    assert np.allclose(energies[-1]*au2cm, std[0], atol=2)
    assert np.allclose(e*au2cm, std[1:4], atol=3)



