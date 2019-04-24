# -*- coding: utf-8 -*-

import logging

import numpy as np
import pytest

from ephMPS.mps import MpDm, Mpo, MpDmFull, SuperLiouville
from ephMPS.transport import ChargeTransport
from ephMPS.utils import Quantity, CompressConfig
from ephMPS.transport.tests.band_param import band_limit_mol_list, low_t, get_analytical_r_square


logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "dissipation, dt, nsteps",
    (
        [0, 4, 20],
        [0.05, 4, 30],
    ),
)
def test_dynamics(dissipation, dt, nsteps):
    tentative_mpo = Mpo(band_limit_mol_list)
    gs_mp = MpDm.max_entangled_gs(band_limit_mol_list)
    # subtract the energy otherwise might cause numeric error because of large offset * dbeta
    energy = Quantity(gs_mp.expectation(tentative_mpo))
    mpo = Mpo(band_limit_mol_list, offset=energy)
    gs_mp = gs_mp.thermal_prop_exact(
        mpo, low_t.to_beta() / 2, 50, "GS", True
    )
    center_mol_idx = band_limit_mol_list.mol_num // 2
    creation_operator = Mpo.onsite(
        band_limit_mol_list, r"a^\dagger", mol_idx_set={center_mol_idx}
    )
    mpdm = creation_operator.apply(gs_mp)
    mpdm_full = MpDmFull.from_mpdm(mpdm)
    # As more compression is involved higher threshold is necessary
    mpdm_full.compress_config = CompressConfig(threshold=1e-4)
    liouville = SuperLiouville(band_limit_mol_list, mpo, dissipation)
    mpdm_full = mpdm
    liouville = mpo
    r_square_list = [mpdm_full.r_square]
    time_series = [0]
    for i in range(nsteps - 1):
        logger.info(mpdm_full)
        mpdm_full = mpdm_full.evolve(liouville, dt)
        r_square_list.append(mpdm_full.r_square)
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


def test_ct():
    ct = ChargeTransport(band_limit_mol_list, dissipation=0.05)
    ct.evolve(4, 30)
    assert (ct.r_square_array[1:] < get_analytical_r_square(np.array(ct.evolve_times))[1:]).all()