import logging
import time

from renormalizer.model import Model, h_qc, basis as ba
from renormalizer.mps import Mps, Mpo, gs
from renormalizer.mps.backend import np
from renormalizer.utils import log

'''
water sto-3g (10e,7o)
O     0.0000000    0.0000000   -0.0644484,
H     0.7499151    0.0000000    0.5114913,
H    -0.7499151    0.0000000    0.5114913,
'''


if __name__ == "__main__":

    start = time.time()
    dump_dir = "./"
    job_name = "qc"  #########
    log.set_stream_level(logging.DEBUG)
    log.register_file_output(dump_dir+job_name+".log", mode="w")
    logger = logging.getLogger(__name__)

    spatial_norbs = 7
    spin_norbs = spatial_norbs * 2
    h1e, h2e, nuc = h_qc.read_fcidump("h2o_fcidump.txt", spatial_norbs) 

    # Potential for H2O has high symmetry and constructed MPO is smaller
    # than MPO in normal case. Use random potential to compare with normal MPO.
    RANDOM_INTEGRAL = False
    if RANDOM_INTEGRAL:
        h1e = np.random.uniform(-1,1,size=(spin_norbs,spin_norbs))
        h2e = np.random.uniform(-1,1,size=(spin_norbs,spin_norbs,spin_norbs,spin_norbs))
        h1e = 0.5*(h1e+h1e.T)
        h2e = 0.5*(h2e+h2e.transpose((2,3,0,1)))

    basis, ham_terms = h_qc.qc_model(h1e, h2e)

    model = Model(basis, ham_terms)
    mpo = Mpo(model)
    logger.info(f"mpo_bond_dims:{mpo.bond_dims}")

    nelec = 10
    energy_list = {}
    M = 50
    procedure = [[M, 0.4], [M, 0.2], [M, 0.1], [M, 0], [M, 0], [M,0], [M,0]]
    mps = Mps.random(model, nelec, M, percent=1.0)

    mps.optimize_config.procedure = procedure
    mps.optimize_config.method = "2site"
    energies, mps = gs.optimize_mps(mps.copy(), mpo)
    gs_e = min(energies)+nuc
    logger.info(f"lowest energy: {gs_e}")
    # fci result
    assert np.allclose(gs_e, -75.008697516450)

    end = time.time()
    logger.info(f"time cost {end - start}")

