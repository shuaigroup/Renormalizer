from renormalizer.model import h_qc, MolList2
from renormalizer.utils import basis as ba
from renormalizer.utils import Op
from renormalizer.cv import batch_run
from renormalizer.cv.zerot import SpectraZtCV
from renormalizer.mps import Mpo, Mps, gs
from renormalizer.cv.tests import cur_dir
import numpy as np
import os


def test_H_chain_LDOS():
    # local density of states of four H_Chain 
    # Ronca,J. Chem. Theory Comput. 2017, 13, 5560-5571
    # example to use Mollist2 to do CV calculation

    spatial_norbs = 4
    spin_norbs = spatial_norbs * 2
    h1e, h2e, nuc = h_qc.read_fcidump(os.path.join(cur_dir,
        "fcidump_lowdin_h4.txt"), spatial_norbs) 
    
    model = h_qc.qc_model(h1e, h2e)

    order = {}
    basis = []
    for iorb in range(spin_norbs):
        order[f"e_{iorb}"] = iorb
        basis.append(ba.BasisHalfSpin(sigmaqn=[0,1]))
    
    mol_list2 = MolList2(order, basis, model)
    mpo = Mpo(mol_list2)
    
    nelec = spatial_norbs
    M = 50
    procedure = [[M, 0.4], [M, 0.2]] + [[M, 0],]*6
    mps = Mps.random(mol_list2, nelec, M, percent=1.0)
    
    mps.optimize_config.procedure = procedure
    mps.optimize_config.method = "2site"
    energies, mps = gs.optimize_mps_dmrg(mps, mpo)
    gs_e = min(energies)+nuc
    
    assert np.allclose(gs_e, -2.190384218792706)
    mps_e = mps.expectation(mpo)
    
    def photoelectron_operator(idx):
        # norbs is the spin orbitals
        # green function 
        model = {}
        key = tuple()
        op = tuple()
        for iorb in range(idx):
            key += (f"e_{iorb}",)   
            op += (Op("sigma_z",0),)
        key += (f"e_{idx}",)
        op += (Op("sigma_+",-1), 1.0)
        model[key] = [op,]
    
        return model

    dipole_model = photoelectron_operator(nelec-1)
    dipole_op = Mpo.general_mpo(mol_list2, model=dipole_model)
    b_mps = dipole_op.apply(mps)

    #std 
    #test_freq = np.linspace(0.25, 1.25, 100, endpoint=False).tolist()
    test_freq = np.linspace(0.25, 1.25, 20, endpoint=False).tolist()
    eta = 0.05
    M = 10
    procedure_cv = [0.4, 0.2] + [0]*6
    spectra = SpectraZtCV(mol_list2, None, M, eta, h_mpo=mpo, method="2site",
            procedure_cv=procedure_cv, b_mps=b_mps.scale(-eta), e0=mps_e)
    
    result = batch_run(test_freq, 1, spectra)
    std = np.load(os.path.join(cur_dir,"H_chain_std.npy"))
    #np.save("res", result)
    #np.save("freq", test_freq)
    assert np.allclose(result, std[::5])
