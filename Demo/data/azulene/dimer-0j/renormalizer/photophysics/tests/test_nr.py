from renormalizer.photophysics import base
from renormalizer.model import Model, Op
from renormalizer.mps import Mps, Mpo, MpDm, ThermalProp
from renormalizer.utils.constant import *
from renormalizer.model import basis as ba
from renormalizer.utils import EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod
from renormalizer.vibronic import VibronicModelDynamics
from renormalizer.utils import log, Quantity

import logging
import itertools 
import numpy as np

logger = logging.getLogger(__name__)

dump_dir = "./"
job_name = "test_d20_m20"  ####################
log.register_file_output(dump_dir+job_name+".log", mode="w")

fdusin = "/home/jjren/Code/reno_std/Renormalizer_photophysics/renormalizer/photophysics/evc.cart.dat"
fnac = "/home/jjren/Code/reno_std/Renormalizer_photophysics/renormalizer/photophysics/evc.cart.nac"
e_ad = 0.0750812420000102
w0, w1, d0, d1, nac, s021, s120 = base.single_mol_model(fdusin, fnac, e_ad, projector=6)
logger.info(f"w: {w1*au2cm}")
logger.info(f"s1 energy: {e_ad+np.sum(w1)/2}")
nmodes = len(w0)

s120 = s120.T
print(s120.dot(s120.T))
assert np.allclose(s120, s021)
# construct the model
ham_terms = []
# kinetic
for imode in range(nmodes):
    ham_terms.append(Op("p^2", f"v_{imode}", factor=1/2, qn=0))

# potential es coordinates
for imode in range(nmodes):
    ham_terms.append(Op("a^\dagger a x^2", ["ex","ex", f"v_{imode}"],
        factor=w1[imode]**2/2, qn=[1,-1,0]))

for imode, jmode in itertools.product(range(nmodes), repeat=2):
    ham_terms.append(Op("a^\dagger a x x", ["gs", "gs", f"v_{imode}",
        f"v_{jmode}"], factor=np.einsum("k,k,k ->", s120[imode,:], w0**2/2,
            s120[jmode,:]), qn=[0,0,0,0]))

ham_terms.append(Op("a^\dagger a", ["gs", "gs"], factor=np.sum(w0**2/2*d0**2),
    qn=[0,0]))

for imode in range(nmodes):
    ham_terms.append(Op("a^\dagger a x", ["gs", "gs", f"v_{imode}"], 
        factor=np.einsum("k,k,k ->", s120[imode,:], w0**2, d0),
        qn=[0,0,0]))

ham_terms.append(Op("a^\dagger a", ["ex","ex"], factor=e_ad, qn=[1,-1]))

basis = []
for imode in range(nmodes):
    basis.append(ba.BasisSHO(f"v_{imode}", w1[imode], 40))
basis.insert(nmodes//2, ba.BasisMultiElectron(["gs","ex"], [0,1]))


para = {"nac":{}}
for imode in range(nmodes):
    para["nac"][f"v_{imode}"] = nac[imode]

model = Model(basis, ham_terms, para=para)

evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
#evolve_config = EvolveConfig()
compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=20)
#compress_config = CompressConfig()
knr = base.ZTnr(model, 1, 0, 
                compress_config=compress_config,
                evolve_config=evolve_config,
                dump_dir=dump_dir, job_name=job_name)
#insteps = 10
#ievolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
#emi = FTnr(model, 1, Quantity(298,"K"), insteps,
#        ievolve_config=ievolve_config, evolve_config=evolve_config,
#        compress_config=compress_config, icompress_config=compress_config)


nsteps = 100
dt = 15.0
knr.evolve(dt, nsteps)
