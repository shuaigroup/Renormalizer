from renormalizer.mps.backend import np, xp
from renormalizer.model.op import Op
from renormalizer.model.basis import BasisHalfSpin
from renormalizer import Model, Mps, Mpo, optimize_mps
from renormalizer.utils import EvolveConfig, EvolveMethod
from renormalizer.utils.log import package_logger as logger

"""
one dimensional Hubbard model with open boundary condition
H = t \sum_{i=1}^{N-1} a_i^\dagger a_i+1 + a_i+1^\dagger a_i + U \sum_i n_{i\arrowup} n_{i\arrowdown}
"""
#------------------------------------------------------------------------
# Jordan-Wigner transformation maps fermion problem into spin problem
#
# |0> => |alpha> and |1> => |beta >:
#
#    a_j   => Prod_{l=0}^{j-1}(sigma_z[l]) * sigma_+[j]
#    a_j^+ => Prod_{l=0}^{j-1}(sigma_z[l]) * sigma_-[j]
# j starts from 0 as in computer science convention to be consistent
# with the following code.
#------------------------------------------------------------------------

nsites = 10
t = -1
U = 4
# the ordering of the spin orbital is
# 0up, 0down, 1up, 1down,...

# the first number of the two-element list is the change of # of alpha
# electrons, the second number is for beta electrons
qn_dict_up = {"+": [-1, 0], "-": [1, 0], "Z": [0, 0]}
qn_dict_do = {"+": [0, -1], "-": [0, 1], "Z": [0, 0]}

ham_terms = []

for i in range(2*(nsites-1)):
    if i % 2 == 0:
        qn1 = [qn_dict_up["Z"], qn_dict_up["+"], qn_dict_do["Z"], qn_dict_up["-"]]
        qn2 = [qn_dict_up["Z"], qn_dict_up["-"], qn_dict_do["Z"], qn_dict_up["+"]]
    else:
        qn1 = [qn_dict_do["Z"], qn_dict_do["+"], qn_dict_up["Z"],
                qn_dict_do["-"]]
        qn2 = [qn_dict_do["Z"], qn_dict_do["-"], qn_dict_up["Z"],
                qn_dict_do["+"]]

    op1 = Op("Z + Z -", [i,i,i+1,i+2], factor=t, qn=qn1)
    op2 = Op("Z - Z +", [i,i,i+1,i+2], factor=-t, qn=qn2)
    ham_terms.extend([op1, op2])

for i in range(0,2*nsites,2):
    qn = [qn_dict_up["-"], qn_dict_up["+"], qn_dict_do["-"], qn_dict_do["+"]]
    op = Op("- + - +", [i,i,i+1,i+1], factor=U, qn=qn)
    ham_terms.append(op)

basis = []
for i in range(2*nsites):
    if i % 2 == 0:
        sigmaqn = np.array([[0, 0], [1, 0]])
    else:
        sigmaqn = np.array([[0, 0], [0, 1]])
    basis.append(BasisHalfSpin(i, sigmaqn=sigmaqn))

model = Model(basis, ham_terms)
mpo = Mpo(model)
logger.info(f"mpo_bond_dims:{mpo.bond_dims}")

nelec = [5, 5]
M = 100
procedure = [[M, 0.4], [M, 0.2], [M, 0.1], [M, 0], [M, 0], [M,0], [M,0]]
mps = Mps.random(model, nelec, M, percent=1.0)
logger.info(f"initial mps: {mps}")

# algorithm 1:  DMRG sweep
mps.optimize_config.procedure = procedure
mps.optimize_config.method = "2site"
energies, mps = optimize_mps(mps.copy(), mpo)
gs_e = min(energies)
logger.info(f"lowest energy: {gs_e}")

# algorithm 2: imaginary time propagation
evolve_config = EvolveConfig(EvolveMethod.tdvp_ps,
        adaptive=True,
        guess_dt=1e-3/1j,
        adaptive_rtol=5e-4,
        ivp_solver="RK45"
        )
mps.evolve_config = evolve_config
evolve_dt = 0.5/1j
energy_old = 0
istep = 0
while True:
    mps = mps.evolve(mpo, evolve_dt)
    energy = mps.expectation(mpo)
    logger.info(f"current mps: {mps}")
    logger.info(f"istep={istep}, energy={energy}")
    if np.abs(energy-energy_old) < 1e-5:
        logger.info("converge!")
        break
    istep += 1
    energy_old = energy
