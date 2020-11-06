# -*- coding: utf-8 -*-

from itertools import product
from typing import List

import numpy as np
import qutip

from renormalizer.model import Op


def get_clist(nsites, ph_levels):
    clist = []
    for i in range(nsites):
        basis = []
        for j in range(nsites):
            if j == i:
                state = qutip.destroy(2)
            else:
                state = qutip.identity(2)
            basis.append(state)
            basis.append(qutip.identity(ph_levels))
        clist.append(qutip.tensor(basis))
    return clist


def get_blist(nsites, ph_levels):
    blist = []
    for i in range(nsites):
        basis = []
        for j in range(nsites):
            basis.append(qutip.identity(2))
            if j == i:
                state = qutip.destroy(ph_levels)
            else:
                state = qutip.identity(ph_levels)
            basis.append(state)
        blist.append(qutip.tensor(basis))
    return blist


def get_holstein_hamiltonian(nsites, J, omega, g, clist, blist, periodic=False):
    lam = g ** 2 * omega
    terms = []
    for i in range(nsites):
        terms.append(lam * clist[i].dag() * clist[i])
        terms.append(omega * blist[i].dag() * blist[i])
        terms.append(-omega * g * clist[i].dag() * clist[i] * (blist[i].dag() + blist[i]))
    hop_limit = nsites if periodic else nsites -1
    for i in range(hop_limit):
        next_i = (i+1) % nsites
        terms.append(J * clist[i].dag() * clist[next_i])
        terms.append(J * clist[i] * clist[next_i].dag())

    return sum(terms)


def get_peierls_hamiltonian(nsites, J, omega, g, clist, blist):
    terms = []
    for i in range(nsites):
        next_i = (i + 1) % nsites
        # electronic coupling
        terms.append(J * clist[i].dag() * clist[next_i])
        terms.append(J * clist[i] * clist[next_i].dag())
        # phonon energy
        terms.append(omega * blist[i].dag() * blist[i])
        # electron-phonon coupling
        terms.append(g * omega * clist[i].dag() * clist[next_i] * (blist[i].dag() + blist[i]))
        terms.append(g * omega * clist[i] * clist[next_i].dag() * (blist[i].dag() + blist[i]))


    return sum(terms)


def get_gs(nsites, ph_levels):
    basis = []
    for i in range(nsites):
        basis.append(qutip.states.basis(2))
        basis.append(qutip.states.basis(ph_levels))
    return qutip.tensor(basis)


def get_qnidx(ph_levels, nsites):
    particles = np.array(list(product(*[[0, 1], [0] * ph_levels] * nsites))).sum(axis=1)
    return np.where(particles == 1)[0]


def get_sigma_list(sigma, nsites):
    sigma_list = []
    for i in range(nsites):
        basis = []
        for j in range(nsites):
            if j == i:
                if sigma == "sigma_+":
                    state = qutip.sigmap()
                elif sigma == "sigma_-":
                    state = qutip.sigmam()
                elif sigma == "sigma_z":
                    state = qutip.sigmaz()
                else:
                    assert False
            else:
                state = qutip.identity(2)
            basis.append(state)
        sigma_list.append(qutip.tensor(basis))
    return sigma_list


def get_spin_hamiltonian(op_terms: List[Op]):
    sigma_dict = {
        "sigma_+": qutip.sigmap(),
        "sigma_-": qutip.sigmam(),
        "sigma_z": qutip.sigmaz()
    }

    terms = []
    for op in op_terms:
        qutip_list = []
        for symbol in op.split_symbol:
            qutip_list.append(sigma_dict[symbol])
        qutip_term = qutip.tensor(qutip_list)
        qutip_term *= op.factor
        terms.append(qutip_term)
    return sum(terms)