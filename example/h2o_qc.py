from renormalizer.mps import Mps, Mpo, gs
from renormalizer.model import MolList2, h_qc
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

start = time.time()
dump_dir = "./"
job_name = "qc"  #########
log.register_file_output(dump_dir+job_name+".log", mode="w")
logger = logging.getLogger(__name__)

spatial_norbs = 7
spin_norbs = spatial_norbs * 2
h1e, h2e, nuc = h_qc.read_fcidump("h2o_fcidump.txt", spatial_norbs) 

# a randon integral
#h1e = np.random.uniform(-1,1,size=(spin_norbs,spin_norbs))
#h2e = np.random.uniform(-1,1,size=(spin_norbs,spin_norbs,spin_norbs,spin_norbs))
#h1e = 0.5*(h1e+h1e.T)
#h2e = 0.5*(h2e+h2e.transpose((2,3,0,1)))

model = h_qc.qc_model(h1e, h2e)

order = {}
basis = []
for iorb in range(spin_norbs):
    order[f"e_{iorb}"] = iorb
    basis.append(ba.BasisHalfSpin(sigmaqn=[0,1]))

mol_list2 = MolList2(order, basis, model)
mpo = Mpo(mol_list2)
logger.info(f"mpo_bond_dims:{mpo.bond_dims}")

nelec = 10
energy_list = {}
M = 50
procedure = [[M, 0.4], [M, 0.2], [M, 0.1], [M, 0], [M, 0], [M,0], [M,0]]
mps = Mps.random(mol_list2, nelec, M, percent=1.0)

mps.optimize_config.procedure = procedure
mps.optimize_config.method = "2site"
energies, mps = gs.optimize_mps_dmrg(mps.copy(), mpo)
gs_e = min(energies)+nuc
logger.info(f"lowest energy: {gs_e}")
# fci result
assert np.allclose(gs_e, -75.008697516450)

end = time.time()
logger.info(f"time cost {end - start}")
