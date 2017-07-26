# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import unittest
from ephMPS import exact_solver
from parameter import *
from ephMPS.constant import *
from ephMPS import nparticle

T = 298.0
eta = 0.00005

class Test_exact_solver(unittest.TestCase):
    
    def test_full_diagonalization(self):
        dyn_omega = np.linspace(1.4, 3.0, num=1000)/au2ev
        
        ix, iy, iph_dof_list, inconfigs = exact_solver.pre_Hmat(0, mol)
        iHmat = exact_solver.construct_Hmat(inconfigs,mol, J,
                indirect=[iph_dof_list, ix, iy])
        ie, ic =  exact_solver.Hmat_diagonalization(iHmat, method="full")
        
        fx, fy, fph_dof_list, fnconfigs = exact_solver.pre_Hmat(1, mol)
        fHmat = exact_solver.construct_Hmat(fnconfigs, mol, J,
                indirect=[fph_dof_list, fx, fy])
        fe, fc =  exact_solver.Hmat_diagonalization(fHmat, method="full")
        
        dipolemat = exact_solver.construct_dipoleMat(inconfigs,fnconfigs,
                mol, indirecti=[iph_dof_list, ix, iy], indirectf=[fph_dof_list, fx, fy])
        dipdip = exact_solver.full_diagonalization_spectrum(ic,ie,fc,fe,dipolemat)
        
        basic_std = np.load("std_data/exact_solver/exact_solver_basic.npz")
        self.assertTrue(np.allclose(ie, basic_std['ie']))
        self.assertTrue(np.allclose(fe, basic_std['fe']))
        self.assertTrue(np.allclose(iHmat.todense(), basic_std['iHmat'].todense()))
        self.assertTrue(np.allclose(fHmat.todense(), basic_std['fHmat'].todense()))
        self.assertTrue(np.allclose(dipolemat.todense(), basic_std['dipolemat'].todense()))

        
        # absorption
        # T = 0
        dyn_corr_absexact = exact_solver.dyn_exact(dipdip, 0, ie)
        spectra_absexact = exact_solver.spectra_normalize(dyn_corr_absexact[1,:])
        with open("std_data/exact_solver/T0abs.npy", 'rb') as f:
            spectra_absexact_std = np.load(f)
        self.assertTrue(np.allclose(spectra_absexact, spectra_absexact_std))

        # T > 0
        dyn_corr1 = exact_solver.dyn_exact(dipdip, T, ie, omega=dyn_omega, eta=eta)
        spectra1 = exact_solver.spectra_normalize(dyn_corr1)
        with open("std_data/exact_solver/TTabs.npy", 'rb') as f:
            spectra1_std = np.load(f)
        self.assertTrue(np.allclose(spectra1, spectra1_std))
        
        # emission
        # T = 0
        dyn_corr_emiexact = exact_solver.dyn_exact(np.transpose(dipdip,(0,2,1)), 0, fe)
        spectra_emiexact = exact_solver.spectra_normalize(dyn_corr_emiexact[1,:])
        with open("std_data/exact_solver/T0emi.npy", 'rb') as f:
            spectra_emiexact_std = np.load(f)
        self.assertTrue(np.allclose(spectra_emiexact, spectra_emiexact_std))
        
        # T > 0
        dyn_corr2 = exact_solver.dyn_exact(np.transpose(dipdip,(0,2,1)), T, fe, \
                omega=dyn_omega, eta=eta)
        spectra2 = exact_solver.spectra_normalize(dyn_corr2)
        with open("std_data/exact_solver/TTemi.npy", 'rb') as f:
            spectra2_std = np.load(f)
        self.assertTrue(np.allclose(spectra2, spectra2_std))
    

    def test_lanczos(self):
        dyn_omega = np.linspace(1.4, 3.0, num=1000)/au2ev
        nsamp = 500
        M = 100
        ix, iy, iph_dof_list, inconfigs = exact_solver.pre_Hmat(0, mol)
        iHmat = exact_solver.construct_Hmat(inconfigs, mol, J,
                indirect=[iph_dof_list, ix, iy])
        fx, fy, fph_dof_list, fnconfigs = exact_solver.pre_Hmat(1, mol)
        fHmat = exact_solver.construct_Hmat(fnconfigs, mol, J,
                indirect=[fph_dof_list, fx, fy])
        dipolemat = exact_solver.construct_dipoleMat(inconfigs,fnconfigs,
                mol, indirecti=[iph_dof_list, ix, iy], indirectf=[fph_dof_list, fx, fy])
        
        # lanczos method
        ie, ic =  exact_solver.Hmat_diagonalization(iHmat, method="Arnoldi")
        self.assertAlmostEqual(ie[0]*au2ev, 0.0)
        
        fe, fc =  exact_solver.Hmat_diagonalization(fHmat, method="Arnoldi")
        self.assertAlmostEqual(fe[0]*au2ev, 2.2861405313)
        
        # absorption
        # T=0
        AiC = exact_solver.dipoleC(mol, ic[:,0], inconfigs, fnconfigs, '+', \
                indirect1=[iph_dof_list, ix, iy], indirect2=[fph_dof_list, fx, fy])
        dyn_corr5 = exact_solver.dyn_lanczos(0.0, dipolemat, iHmat, fHmat, dyn_omega,\
                ie[0], AC=AiC, eta=eta)
        spectra5 = exact_solver.spectra_normalize(dyn_corr5)
        with open("std_data/exact_solver/lanc_T0abs.npy", 'rb') as f:
            spectra5_std = np.load(f)
        self.assertTrue(np.allclose(spectra5, spectra5_std, atol=1e-04))
        
        # T>0
        dyn_corr3 = exact_solver.dyn_lanczos(T, dipolemat, iHmat, fHmat,\
                dyn_omega, ie[0], AC=AiC, eta=eta, nsamp=nsamp, M=M)
        spectra3 = exact_solver.spectra_normalize(dyn_corr3)
        with open("std_data/exact_solver/lanc_TTabs.npy", 'rb') as f:
            spectra3_std = np.load(f)
        self.assertTrue(np.allclose(spectra3, spectra3_std, atol=1e-02))
        
        # emission
        dyn_omega = dyn_omega[::-1] * -1.0
        # T=0
        AfC = exact_solver.dipoleC(mol, fc[:,0], fnconfigs, inconfigs, '-', \
                indirect1=[fph_dof_list, fx, fy], indirect2=[iph_dof_list, ix, iy])
        dyn_corr6 = exact_solver.dyn_lanczos(0.0, dipolemat.T, fHmat, iHmat, dyn_omega,\
                fe[0], AC=AfC, eta=eta)
        spectra6 = exact_solver.spectra_normalize(dyn_corr6)
        with open("std_data/exact_solver/lanc_T0emi.npy", 'rb') as f:
            spectra6_std = np.load(f)
        self.assertTrue(np.allclose(spectra6, spectra6_std, atol=1e-04))
        
        # T>0
        dyn_corr4 = exact_solver.dyn_lanczos(T, dipolemat.T, fHmat, iHmat,\
                dyn_omega, fe[0], AC=AfC, eta=eta, nsamp=nsamp, M=M)
        spectra4 = exact_solver.spectra_normalize(dyn_corr4)
        with open("std_data/exact_solver/lanc_TTemi.npy", 'rb') as f:
            spectra4_std = np.load(f)
        self.assertTrue(np.allclose(spectra4, spectra4_std, atol=1e-02))
    

    def test_n_particle(self):
        dyn_omega = np.linspace(1.4, 3.0, num=1000)/au2ev
        configi_dict, ie = exact_solver.exciton0H(mol, T, 0.00001)
        ic = np.diag([1.0]*len(ie))
        
        # absorption
        # T > 0
        # 1-p
        configf_dict = nparticle.construct_config_dict(mol, 1, nparticle=1)
        fHmat = exact_solver.construct_Hmat(len(configf_dict), mol, J, direct=[nmols, configf_dict])
        fe, fc =  exact_solver.Hmat_diagonalization(fHmat, method="full")
        dipolemat = exact_solver.construct_dipoleMat(len(configi_dict),len(configf_dict),
                mol, directi=[nmols, configi_dict], directf=[nmols, configf_dict])
        dipdip = exact_solver.full_diagonalization_spectrum(ic,ie,fc,fe,dipolemat)
        
        dyn_corr1 = exact_solver.dyn_exact(dipdip, T, ie, omega=dyn_omega, eta=eta)
        spectra1 = exact_solver.spectra_normalize(dyn_corr1)
        with open("std_data/exact_solver/1-p.npy", 'rb') as f:
            spectra1_std = np.load(f)
        self.assertTrue(np.allclose(spectra1, spectra1_std, atol=1e-04))

        # 2-p
        configf_dict = nparticle.construct_config_dict(mol, 1, nparticle=2)
        fHmat = exact_solver.construct_Hmat(len(configf_dict), mol, J, direct=[nmols, configf_dict])
        fe, fc =  exact_solver.Hmat_diagonalization(fHmat, method="full")
        dipolemat = exact_solver.construct_dipoleMat(len(configi_dict),len(configf_dict),
                mol, directi=[nmols, configi_dict], directf=[nmols, configf_dict])
        dipdip = exact_solver.full_diagonalization_spectrum(ic,ie,fc,fe,dipolemat)
        dyn_corr2 = exact_solver.dyn_exact(dipdip, T, ie, omega=dyn_omega, eta=eta)
        spectra2 = exact_solver.spectra_normalize(dyn_corr2)
        with open("std_data/exact_solver/2-p.npy", 'rb') as f:
            spectra2_std = np.load(f)
        self.assertTrue(np.allclose(spectra2, spectra2_std, atol=1e-04))

        # 3-p
        configf_dict = nparticle.construct_config_dict(mol, 1, nparticle=3)
        fHmat = exact_solver.construct_Hmat(len(configf_dict), mol, J, direct=[nmols, configf_dict])
        fe, fc =  exact_solver.Hmat_diagonalization(fHmat, method="full")
        dipolemat = exact_solver.construct_dipoleMat(len(configi_dict),len(configf_dict),
                mol, directi=[nmols, configi_dict], directf=[nmols, configf_dict])
        dipdip = exact_solver.full_diagonalization_spectrum(ic,ie,fc,fe,dipolemat)
        dyn_corr3 = exact_solver.dyn_exact(dipdip, T, ie, omega=dyn_omega, eta=eta)
        spectra3 = exact_solver.spectra_normalize(dyn_corr3)
        with open("std_data/exact_solver/TTabs.npy", 'rb') as f:
            spectra3_std = np.load(f)
        self.assertTrue(np.allclose(spectra3, spectra3_std, atol=1e-04))


if __name__ == "__main__":
    print("Test exact_solver")
    unittest.main()
