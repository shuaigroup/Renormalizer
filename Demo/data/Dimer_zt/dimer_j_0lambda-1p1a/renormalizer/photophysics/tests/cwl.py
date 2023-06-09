from renormalizer.model import Op, Model
from renormalizer.utils import constant, Quantity
from renormalizer.model import basis as ba
from renormalizer.photophysics.base import *
from renormalizer.utils import EvolveConfig, EvolveMethod, Quantity

import numpy as np
import itertools

elocalex = Quantity(3, "eV").as_au()
dipole_abs = 1
nmols = 2

# eV
_j_matrix = (
    np.array([[0.0, 0.0], [0.0, 0.0]]) / constant.au2ev
)

omega = np.array([Quantity(200, "cm^{-1}").as_au(), Quantity(800,
    "cm^{-1}").as_au()])
g_value = np.array([1.0, 1.0])
displacement = g_value / -np.sqrt(omega/2)
ph_phys_dim = [12, 12]

e_reorganization = 0
for iph in range(2):
    e_reorganization += omega[iph]**2 * displacement[iph]**2 * 0.5

basis = []
for imol in range(2):
    for iph in range(2):
        basis.append(ba.BasisSHO(f"v_{imol},{iph}", omega[iph], ph_phys_dim[iph]))
basis.insert(2, ba.BasisMultiElectron(["gs","e_0","e_1"], [0,1,1]))

ham_terms = []
# excitonic coupling
for imol, jmol in itertools.permutations(range(nmols),2):
    ham_terms.append(Op("a^\dagger a", [f"e_{imol}", f"e_{jmol}"],
        factor=_j_matrix[imol,jmol], qn=[1,-1]))

# local excitation
for imol in range(nmols):
    ham_terms.append(Op("a^\dagger a", [f"e_{imol}", f"e_{imol}"],
        factor=elocalex+e_reorganization, qn=[1,-1]))

# harmonic part
for imol in range(2):
    for iph in range(2):
        ham_terms.append(Op("p^2", f"v_{imol},{iph}",
            factor=0.5, qn=0))
        ham_terms.append(Op("x^2", f"v_{imol},{iph}",
            factor=0.5*omega[iph]**2, qn=0))

        # e-ph coupling
        ham_terms.append(Op("a^\dagger a x", [f"e_{imol}", f"e_{imol}",f"v_{imol},{iph}"],
            factor=-omega[iph]**2*displacement[iph], qn=[1,-1,0]))

para = {"dipole":{}}
for imol in range(2):
    para["dipole"][(f"e_{imol}","gs")] = dipole_abs

model = Model(basis, ham_terms, para=para)

evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
abso = ZTabs(model, 0, 0, evolve_config=evolve_config)
#emi = ZTemi(model, 1, 0, evolve_config=evolve_config)
nsteps = 600
dt = 20.0
abso.evolve(dt, nsteps)
np.save("autocorr", abso.autocorr)
np.save("autocorr_time", abso.autocorr_time)
#emi.evolve(dt, nsteps)
#np.save("1mol_autocorr_emi", emi.autocorr)
#np.save("1mol_autocorr_time_emi", emi.autocorr_time)
