# -*- coding: utf-8 -*-

import logging

import numpy as np
import pytest

from renormalizer.mps import MpDm, Mpo, MpDmFull, SuperLiouville, Mps, ThermalProp
from renormalizer.utils import Quantity, CompressConfig
from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.transport.dynamics import calc_r_square
from renormalizer.transport.tests.band_param import band_limit_mol_list, low_t, get_analytical_r_square


logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "dissipation, dt, nsteps",
    (
        [0, 4, 20],
        #[0.05, 4, 30], computational cost too large
    ),
)
def test_dynamics(dissipation, dt, nsteps):
    tentative_mpo = Mpo(band_limit_mol_list)
    gs_mp = MpDm.max_entangled_gs(band_limit_mol_list)
    # subtract the energy otherwise might cause numeric error because of large offset * dbeta
    energy = Quantity(gs_mp.expectation(tentative_mpo))
    mpo = Mpo(band_limit_mol_list, offset=energy)
    tp = ThermalProp(gs_mp, mpo, exact=True, space="GS")
    tp.evolve(None, 50, low_t.to_beta() / 2j)
    gs_mp = tp.latest_mps
    center_mol_idx = band_limit_mol_list.mol_num // 2
    creation_operator = Mpo.onsite(
        band_limit_mol_list, r"a^\dagger", mol_idx_set={center_mol_idx}
    )
    mpdm = creation_operator.apply(gs_mp)
    mpdm_full = MpDmFull.from_mpdm(mpdm)
    # As more compression is involved higher threshold is necessary
    mpdm_full.compress_config = CompressConfig(threshold=1e-4)
    liouville = SuperLiouville(mpo, dissipation)
    r_square_list = [calc_r_square(mpdm_full.e_occupations)]
    time_series = [0]
    for i in range(nsteps - 1):
        logger.info(mpdm_full)
        mpdm_full = mpdm_full.evolve(liouville, dt)
        r_square_list.append(calc_r_square(mpdm_full.e_occupations))
        time_series.append(time_series[-1] + dt)
    time_series = np.array(time_series)
    if dissipation == 0:
        assert np.allclose(get_analytical_r_square(time_series), r_square_list, rtol=1e-2, atol=1e-3)
    else:
        # not much we can do, just basic sanity check
        assert (np.array(r_square_list)[1:] < get_analytical_r_square(time_series)[1:]).all()

#from matplotlib import pyplot as plt
#plt.plot(r_square_list)
#plt.plot(get_analytical_r_square(time_series))
#plt.show()


def test_2site():
    ph = Phonon.simple_phonon(Quantity(1), Quantity(1), 2)
    m = Mol(Quantity(0), [ph])
    mol_list = HolsteinModel([m] * 2, Quantity(1))
    gs_mp = Mpo.onsite(mol_list, opera=r"a^\dagger", mol_idx_set={0}).apply(Mps.ground_state(mol_list, max_entangled=False))
    mpdm = MpDm.from_mps(gs_mp)
    mpdm_full = MpDmFull.from_mpdm(mpdm)
    mpdm_full.compress_config = CompressConfig(threshold=1e-4)
    liouville = SuperLiouville(Mpo(mol_list), dissipation=1)
    ph_occupations_array = []
    energies = []
    for i in range(51):
        logger.info(mpdm_full)
        logger.info(mpdm_full.ph_occupations)
        ph_occupations_array.append(mpdm_full.ph_occupations)
        logger.info(mpdm_full.expectation(liouville))
        energies.append(mpdm_full.expectation(liouville))
        mpdm_full = mpdm_full.evolve(liouville, 0.4)
    ph_occupations_array = np.array(ph_occupations_array)
    assert energies[-1] == pytest.approx(-0.340162 + mol_list.gs_zpe, rel=1e-2)
    assert np.allclose(ph_occupations_array[-1], [0.0930588, 0.099115], rtol=1e-2)


def get_2site_std():
    """
    How to produce the standard data of the test_2site method
    """
    import qutip
    import numpy as np
    from matplotlib import pyplot as plt
    ph_levels = 2
    init_state = qutip.tensor(qutip.basis(2, 1), qutip.basis(2, 0), qutip.basis(ph_levels, 0), qutip.basis(ph_levels, 0))
    c1 = qutip.tensor([qutip.destroy(2), qutip.identity(ph_levels), qutip.identity(2), qutip.identity(ph_levels)])
    c2 = qutip.tensor([qutip.identity(2), qutip.identity(ph_levels), qutip.destroy(2), qutip.identity(ph_levels)])
    b1 = qutip.tensor([qutip.identity(2), qutip.destroy(ph_levels), qutip.identity(2), qutip.identity(ph_levels)])
    b2 = qutip.tensor([qutip.identity(2), qutip.identity(ph_levels), qutip.identity(2), qutip.destroy(ph_levels)])
    J = 1
    omega = 1
    g = -0.707
    lam = g ** 2 * omega
    H = J * c1.dag() * c2 + J * c1 * c2.dag() + lam * c1.dag() * c1 + lam * c2.dag() * c2 + omega * b1.dag() * b1 + omega * b2.dag() * b2 + omega * g * c1.dag() * c1 * (
    b1.dag() + b1) + omega * g * c2.dag() * c2 * (b2.dag() + b2)
    projectors = [b1, b2]
    result = qutip.mesolve(H, init_state, np.linspace(0, 20, 101), c_ops=projectors,
                           e_ops=[c1.dag() * c1, c2.dag() * c2, b1.dag() * b1, b2.dag() * b2, H])
    qutip.plot_expectation_values(result)
    plt.show()