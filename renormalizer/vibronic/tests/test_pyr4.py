import logging
from collections import defaultdict
from itertools import permutations as permut
from itertools import product

import pytest

from renormalizer.model import MolList2
from renormalizer.model.mlist import vibronic_to_general
from renormalizer.mps import Mps, Mpo
from renormalizer.mps.backend import np
from renormalizer.utils import EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod
from renormalizer.utils import Op
from renormalizer.utils import basis as ba
from renormalizer.utils.constant import ev2au, fs2au
from renormalizer.vibronic import VibronicModelDynamics

logger = logging.getLogger(__name__)


def construct_vibronic_model(multi_e, dvr):
    r"""
    Bi-linear vibronic coupling model for Pyrazine, 4 modes
    See: Raab, Worth, Meyer, Cederbaum.  J.Chem.Phys. 110 (1999) 936
    The parameters are from heidelberg mctdh package pyr4+.op
    """
    # frequencies
    w10a = 0.1139 * ev2au
    w6a = 0.0739 * ev2au
    w1 = 0.1258 * ev2au
    w9a = 0.1525 * ev2au

    # energy-gap
    delta = 0.42300 * ev2au

    # linear, on-diagonal coupling coefficients
    # H(1,1)
    _6a_s1_s1 = 0.09806 * ev2au
    _1_s1_s1 = 0.05033 * ev2au
    _9a_s1_s1 = 0.14521 * ev2au
    # H(2,2)
    _6a_s2_s2 = -0.13545 * ev2au
    _1_s2_s2 = 0.17100 * ev2au
    _9a_s2_s2 = 0.03746 * ev2au

    # quadratic, on-diagonal coupling coefficients
    # H(1,1)
    _10a_10a_s1_s1 = -0.01159 * ev2au
    _6a_6a_s1_s1 = 0.00000 * ev2au
    _1_1_s1_s1 = 0.00000 * ev2au
    _9a_9a_s1_s1 = 0.00000 * ev2au
    # H(2,2)
    _10a_10a_s2_s2 = -0.01159 * ev2au
    _6a_6a_s2_s2 = 0.00000 * ev2au
    _1_1_s2_s2 = 0.00000 * ev2au
    _9a_9a_s2_s2 = 0.00000 * ev2au

    # bilinear, on-diagonal coupling coefficients
    # H(1,1)
    _6a_1_s1_s1 = 0.00108 * ev2au
    _1_9a_s1_s1 = -0.00474 * ev2au
    _6a_9a_s1_s1 = 0.00204 * ev2au
    # H(2,2)
    _6a_1_s2_s2 = -0.00298 * ev2au
    _1_9a_s2_s2 = -0.00155 * ev2au
    _6a_9a_s2_s2 = 0.00189 * ev2au

    # linear, off-diagonal coupling coefficients
    _10a_s1_s2 = 0.20804 * ev2au

    # bilinear, off-diagonal coupling coefficients
    # H(1,2) and H(2,1)
    _1_10a_s1_s2 = 0.00553 * ev2au
    _6a_10a_s1_s2 = 0.01000 * ev2au
    _9a_10a_s1_s2 = 0.00126 * ev2au

    model = {}
    e_list = ["s1", "s2"]
    v_list = ["10a", "6a", "9a", "1"]
    for e_idx, e_isymbol in enumerate(e_list):
        for e_jdx, e_jsymbol in enumerate(e_list):
            e_idx_tuple = (f"e_{e_idx}", f"e_{e_jdx}")
            model[e_idx_tuple] = defaultdict(list)
            for v_idx, v_isymbol in enumerate(v_list):
                for v_jdx, v_jsymbol in enumerate(v_list):
                    if v_idx == v_jdx:
                        v_idx_tuple = (f"v_{v_idx}",)
                    else:
                        v_idx_tuple = (f"v_{v_idx}", f"v_{v_jdx}")

                    # linear
                    if v_idx == v_jdx:
                        # if one of the permutation is defined, then the `e_idx_tuple` term should
                        # be defined as required by Hermitian Hamiltonian
                        for eterm1, eterm2 in permut([f"{e_isymbol}", f"{e_jsymbol}"], 2):
                            factor = locals().get(f"_{v_isymbol}_{eterm1}_{eterm2}")
                            if factor is not None:
                                factor *= np.sqrt(eval(f"w{v_isymbol}"))
                                model[e_idx_tuple][v_idx_tuple].append((Op("x", 0), factor))
                                logger.debug(f"term: {v_isymbol}_{e_isymbol}_{e_jsymbol}")
                                break
                            else:
                                logger.debug(f"no term: {v_isymbol}_{e_isymbol}_{e_jsymbol}")

                    # quadratic
                    # use product to guarantee `break` breaks the whole loop
                    it = product(permut([f"{v_isymbol}", f"{v_jsymbol}"], 2),
                                 permut([f"{e_isymbol}", f"{e_jsymbol}"], 2))
                    for (vterm1, vterm2), (eterm1, eterm2) in it:
                        logger.info(f"_{vterm1}_{vterm2}_{eterm1}_{eterm2}")
                        factor = locals().get(f"_{vterm1}_{vterm2}_{eterm1}_{eterm2}")

                        if factor is not None:
                            factor *= np.sqrt(eval(f"w{v_isymbol}") * eval(f"w{v_jsymbol}"))
                            if v_idx == v_jdx:
                                model_term = (Op("x^2", 0), factor)
                            else:
                                model_term = (Op("x", 0), Op("x", 0), factor)
                            model[e_idx_tuple][v_idx_tuple].append(model_term)
                            logger.debug(f"term: {v_isymbol}_{v_jsymbol}_{e_isymbol}_{e_jsymbol}")
                            break
                        else:
                            logger.debug(f"no term: {v_isymbol}_{v_jsymbol}_{e_isymbol}_{e_jsymbol}")

    # electronic coupling
    model[("e_0", "e_0")]["J"] = -delta
    model[("e_1", "e_1")]["J"] = delta

    # vibrational kinetic and potential
    model["I"] = {}
    for v_idx, v_isymbol in enumerate(v_list):
        model["I"][(f"v_{v_idx}",)] = [(Op("p^2", 0), 0.5), (Op("x^2", 0), 0.5 * eval("w" + v_isymbol) ** 2)]

    for e_dof, value in model.items():
        for v_dof, ops in value.items():
            if v_dof == "J":
                logger.info(f"{e_dof},{v_dof},{ops}")
            else:
                for term in ops:
                    logger.info(f"{e_dof},{v_dof},{term}")

    order = {}
    basis = []
    if not multi_e:
        idx = 0
        for e_idx, e_isymbol in enumerate(e_list):
            order[f"e_{e_idx}"] = idx
            basis.append(ba.BasisSimpleElectron())
            idx += 1
    else:
        order = {"e_0": 0, "e_1": 0}
        basis.append(ba.BasisMultiElectron(2, [0, 0]))
        idx = 1

    for v_idx, v_isymbol in enumerate(v_list):
        order[f"v_{v_idx}"] = idx
        basis.append(ba.BasisSHO(locals()[f"w{v_isymbol}"], 30, dvr=dvr))
        idx += 1

    logger.info(f"order:{order}")
    logger.info(f"basis:{basis}")

    return order, basis, model


# todo: vibronic model class
@pytest.mark.parametrize("multi_e, translator, dvr", (
          [False, "vibronic", True],
          [False, "general",  False],
          [True,  "vibronic", False],
          [True,  "general", True],
))
def test_pyr_4mode(multi_e, translator, dvr):

    order, basis, vibronic_model = construct_vibronic_model(multi_e, dvr)
    if translator == "vibronic":
        model = vibronic_model
    elif translator == "general":
        model = vibronic_to_general(vibronic_model)
    else:
        assert False
    mol_list2 = MolList2(order, basis, model)
    mpo = Mpo(mol_list2)
    logger.info(f"mpo_bond_dims:{mpo.bond_dims}")
    # same form whether multi_e is True or False
    init_condition = {"e_1": 1}
    if dvr:
        for dof in mol_list2.v_dofs:
            idx = order[dof]
            init_condition[dof] = basis[idx].dvr_v[0]
    mps = Mps.hartree_product_state(mol_list2, condition=init_condition)

    # for multi-e case the `expand bond dimension` routine is currently not working
    # because creation operator is not defined yet
    mps.use_dummy_qn = True
    mps.build_empty_qn()

    compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=10)

    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    job = VibronicModelDynamics(mol_list2, mps0=mps,
                    h_mpo = mpo,
                    compress_config=compress_config,
                    evolve_config=evolve_config,
                    expand=True)
    time_step_fs = 2
    job.evolve(evolve_dt=time_step_fs * fs2au, nsteps=60)

    from renormalizer.vibronic.tests.mctdh_data import mctdh_data
    assert np.allclose(mctdh_data[::round(time_step_fs/0.5)][:61, 1:],
            job.e_occupations_array, atol=2e-2)
