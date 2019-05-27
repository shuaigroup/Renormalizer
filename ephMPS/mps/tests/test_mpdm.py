# -*- coding: utf-8 -*-

import numpy as np
import pytest

from ephMPS.mps import Mpo, MpDm, MpDmFull, ThermalProp
from ephMPS.model import MolList, Mol, Phonon
from ephMPS.tests import parameter
from ephMPS.utils import Quantity, EvolveConfig


@pytest.mark.parametrize("nmols", (2, 3, 4))
@pytest.mark.parametrize("phonon_freq", (0.01, 0.001, 0.0001))
def test_mpdm_full(nmols, phonon_freq):
    ph = Phonon.simple_phonon(Quantity(phonon_freq), Quantity(1), 2)
    m = Mol(Quantity(0), [ph])
    mol_list = MolList([m] * nmols, Quantity(1))

    gs_dm = MpDm.max_entangled_gs(mol_list)
    beta = Quantity(1000, "K").to_beta()
    tp = ThermalProp(gs_dm, Mpo(mol_list), exact=True, space="GS")
    tp.evolve(None, 50, beta / 1j)
    gs_dm = tp.latest_mps
    assert np.allclose(gs_dm.e_occupations, [0] * nmols)
    e_gs_dm = Mpo.onsite(mol_list, r"a^\dagger", mol_idx_set={0}).apply(gs_dm, canonicalise=True)
    assert np.allclose(e_gs_dm.e_occupations, [1] + [0] * (nmols - 1))

    mpdm_full = MpDmFull.from_mpdm(e_gs_dm)
    assert np.allclose(mpdm_full.e_occupations, e_gs_dm.e_occupations)
    assert np.allclose(mpdm_full.ph_occupations, e_gs_dm.ph_occupations, rtol=1e-3)


@pytest.mark.parametrize(
    "mol_list, etot_std, occ_std, rtol",
    (
        [
            parameter.hybrid_mol_list,
            0.0853441664951,
            [0.20881609, 0.35239430, 0.43878960],
            1e-3,
        ],
        [parameter.mol_list, 0.0853413581416, [0.20881782, 0.35239674, 0.43878545], 5e-3],
    ),
)
@pytest.mark.parametrize("nsteps", (
        100,
        None,
))
def test_thermal_prop(mol_list, etot_std, occ_std, nsteps, rtol):
    init_mps = MpDm.max_entangled_ex(mol_list)
    mpo = Mpo(mol_list)
    beta = Quantity(298, "K").to_beta()
    evolve_time = beta / 2j
    if nsteps is None:
        evolve_config = EvolveConfig(adaptive=True, evolve_dt=beta/100j)
        tp = ThermalProp(init_mps, mpo, evolve_config=evolve_config)
        tp.evolve(evolve_time=evolve_time)
    else:
        tp = ThermalProp(init_mps, mpo)
        tp.evolve(None, nsteps=nsteps, evolve_time=evolve_time)
    mps = tp.latest_mps
    MPO, HAM, Etot, A_el = mps.construct_hybrid_Ham(mpo, debug=True)

    assert np.allclose(Etot, etot_std, rtol=rtol)
    assert np.allclose(A_el, occ_std, rtol=rtol)