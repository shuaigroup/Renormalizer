# -*- coding: utf-8 -*-

import numpy as np

import qutip
import pytest

from renormalizer.model import Op, TI1DModel
from renormalizer.model.basis import BasisSimpleElectron, BasisSHO
from renormalizer.transport.spectral_function import SpectralFunctionZT
from renormalizer.utils import Quantity
from renormalizer.utils.configs import CompressConfig, EvolveMethod, EvolveConfig, CompressCriteria
from renormalizer.utils.qutip_utils import get_clist, get_blist, get_holstein_hamiltonian


def test_spectral_function_bogoliubov():
    # nlevels must be large enough compared with T/omega for the algorithm (Bogoliubov transformation) to work
    # For fast qutip calculation maximum nlevels is 4
    # so temperature has to be low
    temperature = Quantity(0.2)
    nsites = 3
    omega = 1

    nlevels = 4
    g = 1

    ti_basis = [
        BasisSimpleElectron("e"),
        BasisSHO("ph0", omega, nlevels),
        BasisSHO("ph1", omega, nlevels)
    ]
    theta = np.arctanh(np.exp(-temperature.to_beta() * omega / 2))
    ti_local_terms = [
        Op(r"a^\dagger a", "e", g ** 2 * omega),
        Op(r"b^\dagger b", "ph0", omega),
        Op(r"b^\dagger b", "ph1", -omega),
        - g * np.cosh(theta) * omega * Op(r"a^\dagger a", "e") * Op(r"b^\dagger + b", "ph0"),
        - g * np.sinh(theta) * omega * Op(r"a^\dagger a", "e") * Op(r"b^\dagger + b", "ph1")
    ]
    ti_nonlocal_terms = [
        Op(r"a^\dagger a", [(0, "e"), (1, "e")]),
        Op(r"a^\dagger a", [(1, "e"), (0, "e")]),
    ]
    model = TI1DModel(ti_basis, ti_local_terms, ti_nonlocal_terms, nsites)

    compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=24)
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    sf = SpectralFunctionZT(model, compress_config=compress_config, evolve_config=evolve_config)
    sf.evolve(nsteps=5, evolve_time=2.5)

    qutip_res = get_qutip_holstein_sf(nsites, 1, nlevels, omega, g, temperature, sf.evolve_times_array)
    assert np.allclose(sf.G_array[:, 1], qutip_res, rtol=1e-2)


def get_qutip_holstein_sf(nsites, J, ph_levels, omega, g, temperature, time_series):
    if temperature == 0:
        beta = 1e100
    else:
        beta = temperature.to_beta()
    clist = get_clist(nsites, ph_levels)
    blist = get_blist(nsites, ph_levels)

    H = get_holstein_hamiltonian(nsites, J, omega, g, clist, blist, True)
    init_state_list = []
    for i in range(nsites):
        egs = qutip.basis(2, 0)
        init_state_list.append(egs * egs.dag())
        b = qutip.destroy(ph_levels)
        init_state_list.append((-beta * (omega * b.dag() * b)).expm().unit())
    init_state = qutip.tensor(init_state_list)

    return qutip.correlation(H, init_state, [0], time_series, [], clist[1], clist[0].dag())[0] / 1j
