# -*- coding: utf-8 -*-

import pytest
from renormalizer.model import Model, Op
from renormalizer.model import basis as ba
from renormalizer.vibration import Vscf
from renormalizer.vibration.tests import cur_dir

import numpy as np
import os


def test_harmonic_potential():
    w0 = np.load(os.path.join(cur_dir,"w0.npy"))
    nmodes = len(w0)
    
    # construct the model
    ham_terms = []
    # kinetic
    for imode in range(nmodes):
        ham_terms.append(Op("p^2", f"v_{imode}", factor=1/2, qn=0))
    
    # potential es coordinates
    for imode in range(nmodes):
        ham_terms.append(Op("x^2", f"v_{imode}",
            factor=w0[imode]**2/2, qn=0))
    
    basis = []
    for imode in range(nmodes):
        basis.append(ba.BasisSHO(f"v_{imode}", w0[imode], 20))
    
    model = Model(basis, ham_terms)
    scf = Vscf(model)
    scf.kernel()
    for imode in range(nmodes):
        np.testing.assert_allclose(scf.e[imode]-np.sum(w0)/2, w0[imode]*np.arange(20), atol=1e-10)


def test_1mr():
    w0 = np.load(os.path.join(cur_dir,"w0.npy"))
    nmodes = len(w0)
    
    # construct the model
    ham_terms = []
    # kinetic
    for imode in range(nmodes):
        ham_terms.append(Op("p^2", f"v_{imode}", factor=1/2, qn=0))
    
    fname = os.path.join(cur_dir, "prop_no_1.mop")
    # azulene 1MR PES, CJCP, 2021, 34, 565
    for imode in range(nmodes):
        with open(fname) as f:
            lines = f.readlines()
            scale_coeff = float(lines[8].split()[imode])
            poly = {}
            for line in lines:
                dof = line[line.find("(")+1:line.find(")")]
                if dof == f"Q{imode}":
                    order = line[line.find("^")+1:line.find("(")]
                    poly[int(order)] = float(line.split()[0])
        for key, value in poly.items():
            ham_terms.append(Op(f"x^{key}", f"v_{imode}",
                factor=value*scale_coeff**key, qn=0))
    
    basis = []
    for imode in range(nmodes):
        basis.append(ba.BasisSHO(f"v_{imode}", w0[imode], 10))
    
    model = Model(basis, ham_terms)
    scf = Vscf(model)
    scf.kernel()
    vscf_c_1mr = np.load(os.path.join(cur_dir, "vscf_c_1MR.npz"))
    vscf_e_1mr = np.load(os.path.join(cur_dir, "vscf_e_1MR.npz"))

    for imode in range(nmodes):
        # higher excited states are sensitive to numerical errors
        n_states = 4
        for icol in range(n_states):
            wfn1 = scf.c[imode][:,icol]
            wfn2 = vscf_c_1mr[f"arr_{imode}"][:,icol]
            diff = min(np.linalg.norm(wfn1 + wfn2), np.linalg.norm(wfn1 - wfn2))
            np.testing.assert_allclose(diff, 0, atol=1e-2)
        np.testing.assert_allclose(scf.e[imode][:n_states], vscf_e_1mr[f"arr_{imode}"][:n_states], atol=1e-4)

