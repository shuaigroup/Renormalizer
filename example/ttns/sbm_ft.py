import sys

from renormalizer.model import Op
from renormalizer.model import basis as ba
from renormalizer.mps.mps import expand_bond_dimension_general
from renormalizer.sbm import ColeDavidsonSDF
from renormalizer.utils import EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod
from renormalizer.utils import log
from renormalizer.tn import BasisTree, TTNO, TTNS, TreeNodeBasis

import numpy as np


# ita_str = sys.argv[1]  # 010, 025, 100
# omega_c_str = sys.argv[2]  # 001, 100
# beta_str = sys.argv[3]  # 025, 100
# temperature_str = sys.argv[4]   # 05, 10, 15
ita_str = "010"
omega_c_str = "010"
beta_str = "025"
temperature_str = "20"


ita = int(ita_str) / 10 # 1, 2.5, 5, 10
eps = 0
Delta = 1
omega_c = int(omega_c_str) / 10  # 0.1, 1, 10

beta = int(beta_str) / 100  # 0.25, 0.5, 0.75, 1
temperature = int(temperature_str) / 10 # 0.5, 1, 1.5

from renormalizer.utils.log import package_logger
logger = package_logger
dump_dir = "./"
job_name = f"ps1_binary_ita{ita_str}_omega{omega_c_str}_beta{beta_str}_temperature{temperature_str}"  ####################
log.register_file_output(dump_dir+job_name+".log", mode="w")


nmodes = 1000
Ms = 20
upper_limit = 30
sdf = ColeDavidsonSDF(ita, omega_c, beta, upper_limit)

w, c2 = sdf.Wang1(nmodes)
c = np.sqrt(c2)
logger.info(f"w:{w}")
logger.info(f"c:{c}")

reno = sdf.reno(w[-1])
logger.info(f"renormalization constant: {reno}")
Delta *= reno

ham_terms = []

# h_s
ham_terms.extend([Op("sigma_z","spin",factor=eps, qn=0),
        Op("sigma_x","spin",factor=Delta, qn=0)])


# boson energy
for imode in range(nmodes):
    op1 = Op(r"p^2",f"v_{imode}_p",factor=0.5, qn=0)
    op2 = Op(r"x^2",f"v_{imode}_p",factor=0.5*w[imode]**2, qn=0)
    ham_terms.extend([op1,op2])
    op1 = Op(r"p^2",f"v_{imode}_q",factor=-0.5, qn=0)
    op2 = Op(r"x^2",f"v_{imode}_q",factor=-0.5*w[imode]**2, qn=0)
    ham_terms.extend([op1,op2])

theta_array = np.arctanh(np.exp(-w/ temperature / 2))
# system-boson coupling
for imode in range(nmodes):
    theta = theta_array[imode]
    op = Op(r"sigma_z x", ["spin", f"v_{imode}_p"], factor=np.cosh(theta) * c[imode], qn=[0,0])
    ham_terms.append(op)
    op = Op(r"sigma_z x", ["spin", f"v_{imode}_q"], factor=np.sinh(theta) * c[imode], qn=[0,0])
    ham_terms.append(op)


nbas_factor = 2
nbas = np.max([16 * c2/w**3 * np.cosh(theta_array)**2, np.ones(nmodes)*4], axis=0)
nbas = np.min([nbas, np.ones(nmodes)*512], axis=0)
nbas = np.round(nbas).astype(int)
nbas *= nbas_factor
logger.info(nbas)
basis = [ba.BasisHalfSpin("spin",[0,0])]
for imode in range(nmodes):
    basis.append(ba.BasisSHO(f"v_{imode}_p", w[imode], int(nbas[imode])))
    basis.append(ba.BasisSHO(f"v_{imode}_q", w[imode], int(nbas[imode])))


tree_order = 2
basis_vib = basis[1:]
elementary_nodes = []

labels = np.array([[nbas>Ms], [nbas>Ms]]).T.ravel()
root = BasisTree.binary_mctdh(basis_vib, contract_primitive=True, contract_label=labels, dummy_label="n").root

root.add_child(TreeNodeBasis(basis[:1]))

basis_tree = BasisTree(root)
basis_tree.print(print_function=logger.info)

# basis_tree = BasisTree.linear(basis)
ttno = TTNO(basis_tree, ham_terms)
exp_z = TTNO(basis_tree, Op("sigma_z", "spin"))
exp_x = TTNO(basis_tree, Op("sigma_x", "spin"))
ttns = TTNS(basis_tree)
ttns.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=Ms)
ttns = expand_bond_dimension_general(ttns, ttno, ex_mps=None)
logger.info(ttns.bond_dims)
logger.info(ttno.bond_dims)
logger.info(len(ttns))
ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
nsteps = 400
dt = 0.1
expectations = []
for i in range(nsteps):
    ttns = ttns.evolve(ttno, dt)
    z = ttns.expectation(exp_z)
    x = ttns.expectation(exp_x)
    expectations.append((z, x))
    logger.info((z, x))
logger.info(expectations)