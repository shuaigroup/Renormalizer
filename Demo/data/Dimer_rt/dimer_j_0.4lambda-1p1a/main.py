from renormalizer.photophysics import base
from renormalizer.model import Model, Op
from renormalizer.mps import Mps, Mpo, MpDm, ThermalProp
from renormalizer.utils.constant import *
from renormalizer.model import basis as ba
from renormalizer.utils import OptimizeConfig, EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod
from renormalizer.vibronic import VibronicModelDynamics
from renormalizer.utils import log, Quantity,constant

import logging
import itertools 
import numpy as np

logger = logging.getLogger(__name__)

dump_dir = "./"
job_name = "test"  ####################
log.register_file_output(dump_dir+job_name+".log", mode="w")







fdusin = "./par/evc.cart.dat"
fnac = "./par/evc.cart.nac"

# w0, w1, d0, d1, nac, s021, s120 = base.single_mol_model(fdusin, fnac, projector=6)

#d = Quantity(5.52,'eV').as_au()
#alpha = Quantity(3868,'cm^{-1}').as_au()/np.sqrt(2*d)
#e_ad = Quantity(d/4, 'a.u.').as_au()
w0 =np.array([Quantity(1400, "cm^{-1}").as_au()]*2)
w1 =np.array([Quantity(1400, "cm^{-1}").as_au()]*2)

#displacement
HR_factor = 0.5
e_lambda = HR_factor *w1[0]
d = np.sqrt(HR_factor/w1[0]*2)
d1=np.array([0,d])
logger.info(f"d1:{d1}")
e_ad = 10*e_lambda



d0 =d1
nac=np.array([1,0])

nmodes = len(w0)
s021=np.eye(nmodes)
s120= s021

# excitonic coupling
scale =-0.4
reorg_e=np.sum(w1**2*d1**2/2)
logger.info(f"reE: {reorg_e} j: {scale*reorg_e}")
nmols = 2
#lattice =6
j_matrix = np.zeros((nmols,nmols))
#j_matrix[0,1] = reorg_e*scale
#j_matrix[1,0] = reorg_e*scale


j_matrix = np.zeros((nmols,nmols)) + np.diag(np.array([reorg_e *scale]*(nmols-1)),k=1) + np.diag(np.array([reorg_e *scale]*(nmols-1)),k=-1)
#j_matrix[0,nmols-1] = reorg_e*scale
#j_matrix[nmols-1,0] = reorg_e*scale

# T
scale_T = 0.3

"""
for i in range(nmols):
    for j in range(nmols):
        if (j-i == 1 and j%lattice !=0) or (j-i ==-1 and j%lattice !=(lattice-1)) or np.abs(j-i) == lattice:
            j_matrix[i][j] = scale * reorg_e

for i in range(lattice):
    j_matrix[i][nmols-lattice+i] = scale * reorg_e
    j_matrix[nmols-lattice+i][i] = scale *reorg_e
    j_matrix[i*lattice][(i+1)*lattice-1] =scale *reorg_e
    j_matrix[(i+1)*lattice-1][i*lattice] =scale *reorg_e
"""

np.set_printoptions(threshold=np.inf)



logger.info(f"j_matrix: {j_matrix}")
logger.info(f"w: {w1*au2cm}")
logger.info(f"s1 energy_mono: {e_ad+np.sum(w1)/2}")
logger.info(f"s1 energy_agg: {e_ad+np.sum(w1)/2+np.sum(w0)/2*(nmols-1)}")



ham_terms = []





for imol in range(nmols):
    # kinetic
    for imode in range(nmodes):
        ham_terms.append(Op("p^2", f"e_{imol}_v_{imode}", factor=1/2, qn=0))



    #1-mr potenital gs
    # fname = "./par/midas/"
    # for imode in range(nmodes):
    #     with open(fname+"savedir/prop_no_1.mop") as f:
    #         lines = f.readlines()
    #         scale_coeff = float(lines[8].split()[imode])
    #         logger.info(f"scale_coeff:{scale_coeff},omega_cal:{np.sqrt(w0[imode])}")
    #         poly = {}
    #         for line in lines:
    #             dof = line[line.find("(")+1:line.find(")")]
    #             if dof == f"Q{imode}":
    #                 order = line[line.find("^")+1:line.find("(")]
    #                 poly[int(order)] = float(line.split()[0])
    #         logger.info(f"poly:{poly}")
    #     for key, value in poly.items():
    #         ham_terms.append(Op(f"a^\dagger a x^{key}", [f"ex_{imol}",f"ex_{imol}", f"e_{imol}_v_{imode}"],
    #             factor=-1*value*scale_coeff**key, qn=[1,-1,0]))
    #         ham_terms.append(Op(f"x^{key}", [f"e_{imol}_v_{imode}"],
    #             factor=value*scale_coeff**key, qn=[0]))
               
    #ha-potential gs
    for imode in range(nmodes):
       ham_terms.append(Op("a^\dagger a x^2", [f"ex_{imol}",f"ex_{imol}", f"e_{imol}_v_{imode}"],
           factor=-1*w0[imode]**2/2, qn=[1,-1,0]))
       ham_terms.append(Op("x^2", [f"e_{imol}_v_{imode}"],
           factor=w0[imode]**2/2, qn=[0]))


    #ph_e
    for imode, jmode in itertools.product(range(nmodes), repeat=2):
        ham_terms.append(Op("a^\dagger a x x", [f"ex_{imol}", f"ex_{imol}", f"e_{imol}_v_{imode}",
            f"e_{imol}_v_{jmode}"], factor=np.einsum("k,k,k ->", s021[imode,:], w1**2/2,
                s021[jmode,:]), qn=[1,-1,0,0]))

    # lambda
    ham_terms.append(Op("a^\dagger a", [f"ex_{imol}", f"ex_{imol}"], factor=np.sum(w1**2/2*d1**2),
        qn=[1,-1]))

    # ep_coupling
    for imode in range(nmodes):
        ham_terms.append(Op("a^\dagger a x", [f"ex_{imol}", f"ex_{imol}", f"e_{imol}_v_{imode}"],
            factor=np.einsum("k,k,k ->", s021[imode,:], w1**2, d1),
            qn=[1,-1,0]))

    #E_ad
    ham_terms.append(Op("a^\dagger a", [f"ex_{imol}",f"ex_{imol}"], factor=e_ad, qn=[1,-1]))

# exitonic coupling
for imol in range(nmols):
    for jmol in range(nmols):
        ham_terms.append(Op("a^\dagger a", [f"ex_{imol}",f"ex_{jmol}"], factor=j_matrix[imol,jmol], qn=[1, -1]))


#basis
basis = []

#elec_site_name = []
#elec_site_qn =[]
#for imol in range(nmols):
#    elec_site_name=  [f"ex_{imol}"]+ elec_site_name
#    elec_site_qn = [1]+elec_site_qn
#for imol in range(nmols):
#    basis.append(ba.BasisMultiElectronVac([f"ex_{imol}"]))

#basis.append(ba.BasisMultiElectronVac(elec_site_name))



for imol in range(nmols):
    basis.append(ba.BasisMultiElectronVac([f"ex_{imol}"]))
    for imode in range(nmodes):
        basis.append(ba.BasisSHO(f"e_{imol}_v_{imode}", w0[imode], 30))


# basis.insert(nmodes//2, ba.BasisMultiElectron([f"gs_{imol}",f"ex_{imol}"], [0,1]))
#basis.insert(nmodes//2, ba.BasisMultiElectron([f"gs_{imol}",f"ex_{imol}"], [0,1]))

#nac operator
para = {"nac":{}}
for imol in range(nmols):
    for imode in range(nmodes):
        para["nac"][f"e_{imol}_v_{imode}"] = nac[imode]

model = Model(basis, ham_terms, para=para)

nac_terms = []
#for key, value in model.para["nac"].items():
#    nac_terms.append(Op("a partialx", [f"ex_{imol}", key],
#                        factor=-value, qn=[-1, 0]))

#for key, value in model.para["nac"].items():
#    nac_terms.append(Op("I", [f"ex_{imol}"],
#                        factor=1, qn=[0]))


for imol in range(nmols):
    for imode in range(nmodes):
        nac_terms.append(Op("a partialx", [f"ex_{imol}", f"e_{imol}_v_{imode}"],
                            factor=-nac[imode], qn=[-1,0]))



model.mpos["nac_mpo"] = Mpo(model, terms=nac_terms)


# obviouly write nac_op



#eta = 1/(100*cm2au)
sigma = 0.0015
def broad_func(t):
    return np.exp(-t**2*sigma**2/2)
evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
M = 10
optimize_config = OptimizeConfig([[M, 0.5], [M, 0.3], [M, 0.1]] + [[M,0]]*100)
#optimize_config = OptimizeConfig([[M, 0.5], [M, 0.3], [M, 0.1]] + [[M,0]]*1)
M_c = 10
compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=M_c)
stos_compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=100)
#knr = base.ZTnr_state_to_state(broad_func,stos_compress_config, model, 1, 0,
#                optimize_config=optimize_config,
#                compress_config=compress_config,
#                evolve_config=evolve_config,
#                dump_dir=dump_dir, job_name=job_name)


#knr = base.ZTnr(model, 1, 0, 
#                optimize_config=optimize_config,
#                compress_config=compress_config,
#                evolve_config=evolve_config,
#                dump_dir=dump_dir, job_name=job_name)

insteps = 100
ievolve_config = EvolveConfig(EvolveMethod.tdvp_ps,adaptive = True ,guess_dt = 0.001/1j)
knr = base.FTnr(model, 1, Quantity(reorg_e*scale_T,"a.u."), insteps,
       ievolve_config=ievolve_config, evolve_config=evolve_config,
       compress_config=compress_config, icompress_config=compress_config,dump_dir=dump_dir, job_name=job_name)

#insteps = 10
#ievolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
#emi = FTnr(model, 1, Quantity(298,"K"), insteps,
#        ievolve_config=ievolve_config, evolve_config=evolve_config,
#        compress_config=compress_config, icompress_config=compress_config)


nsteps = 100
dt = 0.2 * fs2au
#nsteps = 1000
# nsteps = 800
#dt = 8
knr.evolve(dt, nsteps)
