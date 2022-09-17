# -*- coding: utf-8 -*-

import numpy as np
import pytest
import qutip

from renormalizer.model import Phonon, Mol, HolsteinModel, Model
from renormalizer.model.basis import BasisSimpleElectron, BasisSHO
from renormalizer.model.op import Op
from renormalizer.mps.backend import backend
from renormalizer.transport.kubo import TransportKubo
from renormalizer.utils import Quantity, CompressConfig, EvolveConfig, EvolveMethod, CompressCriteria
from renormalizer.utils.qutip_utils import get_clist, get_blist, get_holstein_hamiltonian, get_qnidx, \
    get_peierls_hamiltonian


@pytest.mark.parametrize("scheme", (
        3,
        4,
))
def test_holstein_kubo(scheme):
    ph = Phonon.simple_phonon(Quantity(1), Quantity(1), 2)
    mol = Mol(Quantity(0), [ph])
    model = HolsteinModel([mol] * 5, Quantity(1), scheme)
    temperature = Quantity(50000, 'K')
    compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=24)
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, adaptive=True, guess_dt=0.5, adaptive_rtol=1e-3)
    ievolve_config = EvolveConfig(EvolveMethod.tdvp_ps, adaptive=True, guess_dt=-0.1j)
    kubo = TransportKubo(model, temperature, compress_config=compress_config, ievolve_config=ievolve_config, evolve_config=evolve_config)
    kubo.evolve(nsteps=5, evolve_time=5)
    qutip_res = get_qutip_holstein_kubo(model, temperature, kubo.evolve_times_array)
    rtol = 5e-2
    assert np.allclose(kubo.auto_corr, qutip_res, rtol=rtol)


def get_qutip_holstein_kubo(model, temperature, time_series):

    nsites = len(model)
    J = model.j_constant
    ph = model[0].ph_list[0]
    ph_levels = ph.n_phys_dim
    omega = ph.omega[0]
    g = - ph.coupling_constant
    clist = get_clist(nsites, ph_levels)
    blist = get_blist(nsites, ph_levels)

    qn_idx = get_qnidx(ph_levels, nsites)
    H = get_holstein_hamiltonian(nsites, J, omega, g, clist, blist).extract_states(qn_idx)
    init_state = (-temperature.to_beta() * H).expm().unit()

    terms = []
    for i in range(nsites - 1):
        terms.append(J * clist[i].dag() * clist[i + 1])
        terms.append(-J * clist[i] * clist[i + 1].dag())
    j_oper = sum(terms).extract_states(qn_idx)

    # Add the negative sign because j is taken to be real
    return -qutip.correlation(H, init_state, [0], time_series, [], j_oper, j_oper)[0]


def test_peierls_kubo():
    if backend.is_32bits:
        pytest.skip("VMF too stiff for 32 bit float point operation")
    # number of mol
    n = 4
    # electronic coupling
    V = -Quantity(120, "meV").as_au()
    # intermolecular vibration freq
    omega = Quantity(50, "cm-1").as_au()
    # intermolecular coupling constant
    g = 4
    # number of quanta
    nlevels = 2
    # temperature
    temperature = Quantity(300, "K")

    # the Peierls model
    ham_terms = []
    for i in range(n):
        i1, i2 = i, (i+1) % n
        # H_e
        hop1 = Op(r"a^\dagger a", [i1, i2], V)
        hop2 = Op(r"a a^\dagger", [i1, i2], V)
        ham_terms.extend([hop1, hop2])
        # H_ph
        ham_terms.append(Op(r"b^\dagger b", (i, 0), omega))
        # H_(e-ph)
        coup1 = Op(r"b^\dagger + b", (i, 0)) * Op(r"a^\dagger a", [i1, i2]) * g * omega
        coup2 = Op(r"b^\dagger + b", (i, 0)) * Op(r"a a^\dagger", [i1, i2]) * g * omega
        ham_terms.extend([coup1, coup2])

    basis = []
    for ni in range(n):
        basis.append(BasisSimpleElectron(ni))
        basis.append(BasisSHO((ni, 0), omega, nlevels))

    model = Model(basis, ham_terms)
    compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=24)
    ievolve_config = EvolveConfig(EvolveMethod.tdvp_vmf, ivp_atol=1e-3, ivp_rtol=1e-5)
    evolve_config = EvolveConfig(EvolveMethod.tdvp_vmf, ivp_atol=1e-3, ivp_rtol=1e-5)
    kubo = TransportKubo(model, temperature, compress_config=compress_config, ievolve_config=ievolve_config, evolve_config=evolve_config)
    kubo.evolve(nsteps=5, evolve_time=1000)

    qutip_corr, qutip_corr_decomp = get_qutip_peierls_kubo(V, n, nlevels, omega, g, temperature, kubo.evolve_times_array)
    atol = 1e-7
    rtol = 5e-2
    # direct comparison may fail because of different sign
    assert np.allclose(kubo.auto_corr, qutip_corr, atol=atol, rtol=rtol)
    assert np.allclose(kubo.auto_corr_decomposition, qutip_corr_decomp, atol=atol, rtol=rtol)


def get_qutip_peierls_kubo(J, nsites, ph_levels, omega, g, temperature, time_series):
    clist = get_clist(nsites, ph_levels)
    blist = get_blist(nsites, ph_levels)

    qn_idx = get_qnidx(ph_levels, nsites)
    H = get_peierls_hamiltonian(nsites, J, omega, g, clist, blist).extract_states(qn_idx)
    init_state = (-temperature.to_beta() * H).expm().unit()

    holstein_terms = []
    peierls_terms = []
    for i in range(nsites):
        next_i = (i + 1) % nsites
        holstein_terms.append( J * clist[i].dag() * clist[next_i])
        holstein_terms.append(-J * clist[i] * clist[next_i].dag())
        peierls_terms.append( g * omega * clist[i].dag() * clist[next_i] * (blist[i].dag() + blist[i]))
        peierls_terms.append(-g * omega * clist[i] * clist[next_i].dag() * (blist[i].dag() + blist[i]))
    j_oper1 = sum(holstein_terms).extract_states(qn_idx)
    j_oper2 = sum(peierls_terms).extract_states(qn_idx)

    # Add negative signs because j is taken to be real
    corr1 = -qutip.correlation(H, init_state, [0], time_series, [], j_oper1, j_oper1)[0]
    corr2 = -qutip.correlation(H, init_state, [0], time_series, [], j_oper1, j_oper2)[0]
    corr3 = -qutip.correlation(H, init_state, [0], time_series, [], j_oper2, j_oper1)[0]
    corr4 = -qutip.correlation(H, init_state, [0], time_series, [], j_oper2, j_oper2)[0]
    corr = corr1 + corr2 + corr3 + corr4
    return corr, np.array([corr1, corr2, corr3, corr4]).T
