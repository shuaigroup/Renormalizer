from renormalizer.mps import Mps, Mpo, solver
from renormalizer.model import MolList2, ModelTranslator
from renormalizer.utils import basis as ba
from renormalizer.utils import Op
from renormalizer.utils import log

import numpy as np
import itertools
from collections import defaultdict
import logging
import time
import itertools

'''
water sto-3g (10e,7o)
O     0.0000000    0.0000000   -0.0644484,
H     0.7499151    0.0000000    0.5114913,
H    -0.7499151    0.0000000    0.5114913,
'''

def read_fcidump(fname, norb):
    eri = np.zeros((norb, norb, norb, norb))
    h = np.zeros((norb,norb))
    
    with open(fname, "r") as f:
        a = f.readlines()
        for line, info in enumerate(a):
            if line < 4:
                continue
            s  = info.split()
            integral, p, q, r, s = float(s[0]),int(s[1]),int(s[2]),int(s[3]),int(s[4])
            if r != 0:
                eri[p-1,q-1,r-1,s-1] = integral
                eri[q-1,p-1,r-1,s-1] = integral
                eri[p-1,q-1,s-1,r-1] = integral
                eri[q-1,p-1,s-1,r-1] = integral
            elif p != 0:
                h[p-1,q-1] = integral
                h[q-1,p-1] = integral
            else:
                nuc = integral
    
    nsorb = norb*2
    seri = np.zeros((nsorb, nsorb, nsorb, nsorb))
    sh = np.zeros((nsorb,nsorb))
    for p, q, r, s in itertools.product(range(nsorb),repeat=4):
    # a_p^\dagger a_q^\dagger a_r a_s
        if p % 2 == s % 2 and q % 2 == r % 2:
            seri[p,q,r,s] = eri[p//2,s//2,q//2,r//2] 
    
    for q, s in itertools.product(range(nsorb),repeat=2):
        if q % 2 == s % 2:
            sh[q,s] = h[q//2,s//2]
    
    aseri = np.zeros((nsorb, nsorb, nsorb, nsorb))
    for q, s in itertools.product(range(nsorb),repeat=2):
        for p, r in itertools.product(range(q), range(s)):
            #aseri[p,q,r,s] = seri[p,q,r,s] - seri[q,p,r,s]
            aseri[p,q,r,s] = seri[p,q,r,s] - seri[p,q,s,r]
    
    logger.info(f"nuclear repulsion: {nuc}")
    
    return sh, aseri, nuc


def qc_model(h1e, h2e):
    #------------------------------------------------------------------------
    # Jordan-Wigner transformation maps fermion problem into spin problem
    #
    # |0> => |alpha> and |1> => |beta >: 
    #
    #    a_j^+ => Prod_{l=1}^{j-1}(sigma_z[l]) * sigma_-[j]
    #    a_j   => Prod_{l=1}^{j-1}(sigma_z[l]) * sigma_+[j] 
    #------------------------------------------------------------------------
    
    norbs = h1e.shape[0]
    logger.info(f"spin norbs: {norbs}")
    assert np.all(np.array(h1e.shape) == norbs)
    assert np.all(np.array(h2e.shape) == norbs)

    model = defaultdict(list)
    
    # sigma_-, sigma_+, sigma_z
    # one electron
    for idxs in itertools.product(range(norbs),repeat = 2):
        sort = np.argsort(idxs)
        
        key = tuple()
        op = tuple()
        line1 = []
        line2 = []
        for idx in range(idxs[sort[0]],idxs[sort[1]]+1):
            key += (f"e_{idx}",)
            if idx == idxs[0]:
                line1.append(("sigma_-",1))
            elif idx < idxs[0]:
                line1.append(("sigma_z",0))
            else:
                line1.append(("",0))
    
            if idx == idxs[1]:
                line2.append(("sigma_+",-1))
            elif idx < idxs[1]:
                line2.append(("sigma_z",0))
            else:
                line2.append(("",0))
        
        npermute = 0
        for term1, term2 in zip(line1, line2):
            ops = list(filter(lambda a: a != "", [term1[0],term2[0]]))
            
            sz_idx = [i for i,j in enumerate(ops) if j == "sigma_z"]
            for index, i in enumerate(sz_idx):
                npermute += i - len(sz_idx[:index])
            ops = list(filter(lambda a: a != "sigma_z", ops))
            
            ops = " ".join(ops)
            if len(sz_idx) % 2 == 1:
                ops = ("sigma_z " + ops).strip()
    
            if ops == "":
                ops = "I"
            
            op += (Op(ops, term1[1]+term2[1]),)
    
        #print(idxs)
        #print(key)
        #print(op)
        op += (h1e[idxs[0], idxs[1]]*(-1)**(npermute%2),)
        model[key].append(op)
    
    #2e term
    for q,s in itertools.product(range(norbs),repeat = 2):
        # a^\dagger_p a^\dagger_q a_r a_s
        for p,r in itertools.product(range(q),range(s)):
            idxs = [p,q,r,s]
            sort = np.argsort(idxs)
            
            key = tuple()
            op = tuple()
            line1 = []
            line2 = []
            line3 = []
            line4 = []
    
            for idx in range(idxs[sort[0]],idxs[sort[3]]+1):
                key += (f"e_{idx}",)
                if idx == p:
                    line1.append(("sigma_-",1))
                elif idx < p:
                    line1.append(("sigma_z",0))
                else:
                    line1.append(("",0))
    
                if idx == q:
                    line2.append(("sigma_-",1))
                elif idx < q:
                    line2.append(("sigma_z",0))
                else:
                    line2.append(("",0))
                
                if idx == r:
                    line3.append(("sigma_+",-1))
                elif idx < r:
                    line3.append(("sigma_z",0))
                else:
                    line3.append(("",0))
                
                if idx == s:
                    line4.append(("sigma_+",-1))
                elif idx < s:
                    line4.append(("sigma_z",0))
                else:
                    line4.append(("",0))
    
            
            npermute = 0
            for term1, term2, term3, term4 in zip(line1, line2, line3, line4):
                ops = [op for op in [term1[0],term2[0],term3[0],term4[0]] if op != ""]
                sz_idx = [i for i,j in enumerate(ops) if j == "sigma_z"]
                #for index, i in enumerate(sz_idx):
                #    npermute += i - len(sz_idx[:index])
                nn = len(sz_idx)
                npermute += sum(sz_idx) - int((nn-1)*nn/2)
                ops = [op for op in ops if op != "sigma_z"]
                
                ops = " ".join(ops)
                if nn % 2 == 1:
                    ops = ("sigma_z " + ops).strip()
    
                if ops == "":
                    ops = "I"
                op += (Op(ops, term1[1]+term2[1]+term3[1]+term4[1]),)
    
            #print(p,q,r,s)
            #print(key)
            #print(op,(-1)**(npermute%2), npermute)
            op += (h2e[p,q,r,s]*(-1)**(npermute%2),)
            model[key].append(op)
    
    return model



start = time.time()
dump_dir = "./"
job_name = "qc"  #########
log.register_file_output(dump_dir+job_name+".log", mode="w")
logger = logging.getLogger(__name__)

spatial_norbs = 7
spin_norbs = spatial_norbs * 2
h1e, h2e, nuc = read_fcidump("h2o_fcidump.out", spatial_norbs) 

# a randon integral
#h1e = np.random.uniform(-1,1,size=(spin_norbs,spin_norbs))
#h2e = np.random.uniform(-1,1,size=(spin_norbs,spin_norbs,spin_norbs,spin_norbs))
#h1e = 0.5*(h1e+h1e.T)
#h2e = 0.5*(h2e+h2e.transpose((2,3,0,1)))

model = qc_model(h1e, h2e)

order = {}
basis = []
for iorb in range(spin_norbs):
    order[f"e_{iorb}"] = iorb
    basis.append(ba.BasisHalfSpin(sigmaqn=[0,1]))

mol_list2 = MolList2(order, basis, model, ModelTranslator.general_model)
mpo = Mpo(mol_list2)
logger.info(f"mpo_bond_dims:{mpo.bond_dims}")

nelec = 10
energy_list = {}
M = 50
procedure = [[M, 0.4], [M, 0.2], [M, 0.1], [M, 0], [M, 0], [M,0], [M,0]]
mps = Mps.random(mol_list2, nelec, M, percent=1.0)

mps.optimize_config.procedure = procedure
mps.optimize_config.method = "2site"
energies = solver.optimize_mps_dmrg(mps.copy(), mpo)
gs_e = energies.min()+nuc
logger.info(f"lowest energy: {gs_e}")
# fci result
assert np.allclose(gs_e, -75.008697516450)

end = time.time()
logger.info(f"time cost {end - start}")
