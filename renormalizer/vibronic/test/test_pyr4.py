from renormalizer.utils.constant import *
from renormalizer.utils import Op
from renormalizer.mps import Mpo, Mps
from renormalizer.model import MolList2, ModelTranslator
from renormalizer.utils import basis as ba
from renormalizer.vibronic import VibronicModelDynamics
from renormalizer.utils import EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod
from renormalizer.mps import Mps, Mpo

import logging
import numpy as np
import itertools
import pytest
from collections import defaultdict

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("multi_e", (False, True))
def test_vibronic_model(multi_e):
    r"""
    Bi-linear vibronic coupling model for Pyrazine, 4 modes
    See: Raab, Worth, Meyer, Cederbaum.  J.Chem.Phys. 110 (1999) 936
    The parameters are from heidelberg mctdh package pyr4+.op
    """
    # frequencies
    w10a  = 0.1139  * ev2au
    w6a   = 0.0739  * ev2au
    w1    = 0.1258  * ev2au
    w9a   = 0.1525  * ev2au
    
    # energy-gap
    delta =   0.42300 * ev2au
    
    # linear, on-diagonal coupling coefficients
    # H(1,1)
    _6a_s1_s1  =   0.09806 * ev2au
    _1_s1_s1  =   0.05033 * ev2au
    _9a_s1_s1  =   0.14521 * ev2au
    # H(2,2)
    _6a_s2_s2  =  -0.13545 * ev2au
    _1_s2_s2  =   0.17100 * ev2au
    _9a_s2_s2  =   0.03746 * ev2au
    
    # quadratic, on-diagonal coupling coefficients
    # H(1,1)
    _10a_10a_s1_s1  =  -0.01159 * ev2au
    _6a_6a_s1_s1  =   0.00000 * ev2au
    _1_1_s1_s1  =   0.00000 * ev2au
    _9a_9a_s1_s1  =  0.00000 * ev2au
    # H(2,2)
    _10a_10a_s2_s2 =  -0.01159 * ev2au
    _6a_6a_s2_s2  =  0.00000 * ev2au
    _1_1_s2_s2  =   0.00000 * ev2au
    _9a_9a_s2_s2  =   0.00000 * ev2au
    
    # bilinear, on-diagonal coupling coefficients
    # H(1,1)
    _6a_1_s1_s1  =   0.00108 * ev2au
    _1_9a_s1_s1  =  -0.00474 * ev2au
    _6a_9a_s1_s1  =   0.00204 * ev2au
    # H(2,2)
    _6a_1_s2_s2  =  -0.00298 * ev2au
    _1_9a_s2_s2  =  -0.00155 * ev2au
    _6a_9a_s2_s2  =   0.00189 * ev2au
    
    # linear, off-diagonal coupling coefficients
    _10a_s1_s2 =   0.20804 * ev2au
    
    # bilinear, off-diagonal coupling coefficients
    # H(1,2) and H(2,1)
    _1_10a_s1_s2  =   0.00553 * ev2au
    _6a_10a_s1_s2 =  0.01000 * ev2au
    _9a_10a_s1_s2  =   0.00126 * ev2au
    
    
    model = {}
    e_list = ["s1","s2"]
    v_list = ["10a","6a","9a","1"]
    for e_idx, e_isymbol in enumerate(e_list):
        for e_jdx, e_jsymbol in enumerate(e_list):
            model[(f"e_{e_idx}", f"e_{e_jdx}")] = {}
            for v_idx, v_isymbol in enumerate(v_list):
                for v_jdx, v_jsymbol in enumerate(v_list):
                    #v_idx, v_jdx = v_isymbol, v_jsymbol 
    
                    if v_idx == v_jdx:
                        model[(f"e_{e_idx}", f"e_{e_jdx}")][(f"v_{v_idx}",)] = []
                    else:
                        model[(f"e_{e_idx}", f"e_{e_jdx}")][(f"v_{v_idx}",f"v_{v_jdx}")] = []
                    
                    # linear
                    if v_idx == v_jdx:
                        factor = None
                        for eterm1, eterm2 in \
                            itertools.permutations([f"{e_isymbol}", f"{e_jsymbol}"], 2): 
                            try:
                                factor = eval(f"_{v_isymbol}_{eterm1}_{eterm2}")
                            except:
                                pass
                        if factor is not None:
                            factor *= np.sqrt(eval(f"w{v_isymbol}"))
                            model[(f"e_{e_idx}", f"e_{e_jdx}")][(f"v_{v_idx}",)].append((Op("x",0), factor))
                            logger.debug(f"term: {v_isymbol}_{e_isymbol}_{e_jsymbol}")
                        else:
                            logger.debug(f"no term: {v_isymbol}_{e_isymbol}_{e_jsymbol}")
                    
                    # quadratic
                    factor = None
                    for vterm1, vterm2 in \
                        itertools.permutations([f"{v_isymbol}", f"{v_jsymbol}"], 2): 
                        for eterm1, eterm2 in \
                            itertools.permutations([f"{e_isymbol}", f"{e_jsymbol}"], 2): 
                            try:
                                logger.info(f"_{vterm1}_{vterm2}_{eterm1}_{eterm2}")
                                factor = eval(f"_{vterm1}_{vterm2}_{eterm1}_{eterm2}")
                            except:
                                pass
    
                    if factor is not None:
                        factor *= np.sqrt(eval(f"w{v_isymbol}")*eval(f"w{v_jsymbol}"))
                        if v_idx == v_jdx:
                            model[(f"e_{e_idx}",
                                f"e_{e_jdx}")][(f"v_{v_idx}",)].append(
                                        (Op("x^2",0),factor))
                        else:
                            model[(f"e_{e_idx}",
                                f"e_{e_jdx}")][(f"v_{v_idx}",f"v_{v_jdx}")].append(
                                        (Op("x",0), Op("x",0),factor))
                        logger.debug(f"term: {v_isymbol}_{v_jsymbol}_{e_isymbol}_{e_jsymbol}")
                    else:
                        logger.debug(f"no term: {v_isymbol}_{v_jsymbol}_{e_isymbol}_{e_jsymbol}")
    
    
    # electronic coupling
    model[("e_0", "e_0")]["J"] = -delta 
    model[("e_1", "e_1")]["J"] = delta 
    
    # vibrational kinetic and potential
    model["I"] = {}
    for v_idx, v_isymbol in enumerate(v_list):
        model["I"][(f"v_{v_idx}",)] = [(Op("p^2",0), 0.5),(Op("x^2",0),0.5*eval("w"+v_isymbol)**2)]
    
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
            basis.append(ba.Basis_Simple_Electron())
            idx += 1
    else:
        order = {"e_0":0, "e_1":0}
        basis.append(ba.Basis_Multi_Electron(2,[0,0]))
        idx = 1
    
    for v_idx, v_isymbol in enumerate(v_list):
        order[f"v_{v_idx}"] = idx
        basis.append(ba.Basis_SHO(eval(f"w{v_isymbol}"), 30))
        idx += 1
    
    logger.info(f"order:{order}")
    logger.info(f"basis:{basis}")
    
    mol_list2 = MolList2(order, basis, model, ModelTranslator.vibronic_model)
    mpo = Mpo(mol_list2)
    logger.info(f"mpo_bond_dims:{mpo.bond_dims}")
    mps = Mps.hartree_product_state(mol_list2,condition={"e_1":1})
    print(mps.qn, mps.qntot, mps.qnidx)
    
    compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=10)
    
    evolve_config = EvolveConfig(EvolveMethod.prop_and_compress)
    job = VibronicModelDynamics(mol_list2, mps0=mps,
                    h_mpo = mpo, 
                    compress_config=compress_config,
                    evolve_config=evolve_config)
    job.evolve(evolve_dt=5.0, nsteps=3)
    
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    job.evolve_config = evolve_config
    job.latest_mps.evolve_config = evolve_config
    job.evolve(evolve_dt=80.0, nsteps=60)

    std = np.load("std.npz")
    assert np.allclose(std['electron occupations array'], job.e_occupations_array)

@pytest.mark.parametrize("multi_e", (False, True))
def test_general_model(multi_e):
    
    # frequencies
    w10a  = 0.1139  * ev2au
    w6a   = 0.0739  * ev2au
    w1    = 0.1258  * ev2au
    w9a   = 0.1525  * ev2au
    
    # energy-gap
    delta =   0.42300 * ev2au
    
    coup = {\
            "6a"    :         1.,
            "9a"    :         1.,
            "1"     :         1.,
            "10a"   :         1.,
            "s1_s1" :     -delta,
            "s2_s2" :     delta,
    # linear, on-diagonal coupling coefficients
    # H(1,1)
            "6a_s1_s1"  :   0.09806 * ev2au,
            "1_s1_s1"   :   0.05033 * ev2au,
            "9a_s1_s1"  :   0.14521 * ev2au,
    # H(2,2)
            "6a_s2_s2"  :  -0.13545 * ev2au,
            "1_s2_s2"   :   0.17100 * ev2au,
            "9a_s2_s2"  :   0.03746 * ev2au,
    # quadratic, on-diagonal coupling coefficients
    # H(1,1)
        "10a_10a_s1_s1" :  -0.01159 * ev2au,
        "6a_6a_s1_s1"   :   0.00000 * ev2au,
        "1_1_s1_s1"     :   0.00000 * ev2au,
        "9a_9a_s1_s1"   :   0.00000 * ev2au,
    # H(2,2)
        "10a_10a_s2_s2" :  -0.01159 * ev2au,
        "6a_6a_s2_s2"   :   0.00000 * ev2au,
        "1_1_s2_s2"     :   0.00000 * ev2au,
        "9a_9a_s2_s2"   :   0.00000 * ev2au,
    # bilinear, on-diagonal coupling coefficients
    # H(1,1)
        "6a_1_s1_s1"    :   0.00108 * ev2au,
        "1_9a_s1_s1"    :  -0.00474 * ev2au,
        "6a_9a_s1_s1"   :   0.00204 * ev2au,
    # H(2,2)
        "6a_1_s2_s2"    :  -0.00298 * ev2au,
        "1_9a_s2_s2"    :  -0.00155 * ev2au,
        "6a_9a_s2_s2"   :   0.00189 * ev2au,
    # linear, off-diagonal coupling coefficients
        "10a_s1_s2"     :   0.20804 * ev2au,
    # bilinear, off-diagonal coupling coefficients
    # H(1,2) and H(2,1)
        "1_10a_s1_s2"   :   0.00553 * ev2au,
        "6a_10a_s1_s2"  :   0.01000 * ev2au,
        "9a_10a_s1_s2"  :   0.00126 * ev2au,
        }
    
    model = defaultdict(list)
    
    alias = {"10a":"v_0", "6a":"v_1", "9a":"v_2", "1":"v_3", "s1":"e_0", "s2":"e_1"}
    
    for key, value in coup.items():
        idx = key.split("_")
        if len(idx) == 1:
            model[(alias[idx[0]],)].append((Op(r"p^2", 0), 0.5))
            model[(alias[idx[0]],)].append((Op(r"x^2", 0), 0.5*eval(f"w{idx[0]}")**2))
            
        elif len(idx) == 2:
            # electronic term
            factor = value
            if idx[0] == idx[1]:   # diagonal
                model[(alias[idx[0]],)].append((Op(r"a^\dagger a", 0), factor))
            else:  # off-diagonal
                model[tuple([alias[i] for i in idx])].append((Op(r"a^\dagger", 1),
                    Op("a", -1), factor))
                model[tuple([alias[i] for i in idx])].append((Op(r"a", -1),
                    Op(r"a^\dagger", 1), factor))
    
        elif len(idx) == 3:
            # linear term
            factor = value * np.sqrt(eval(f"w{idx[0]}"))
            if idx[1] == idx[2]: # diagonal
                model[tuple([alias[i] for i in idx[:-1]])].append((Op("x", 0),
                    Op(r"a^\dagger a", 0), factor))
            else:  # off-diagonal
                model[tuple([alias[i] for i in idx])].append((Op("x", 0),
                    Op(r"a^\dagger", 1), Op("a", -1), factor))
                model[tuple([alias[i] for i in idx])].append((Op("x", 0), Op("a", -1),
                    Op(r"a^\dagger", 1), factor))
    
        elif len(idx) == 4:
            # quadratic term
            if idx[0] != idx[1]:  #bilinear
                factor = 2*value
                vterm = (alias[idx[0]], alias[idx[1]])
                vop = (Op("x", 0), Op("x", 0))
            else: # quadractic
                factor = value
                vterm = (alias[idx[0]], )
                vop = (Op("x^2", 0),)
            factor *= np.sqrt(eval(f"w{idx[0]}")*eval(f"w{idx[1]}"))
    
            if idx[2] == idx[3]: # diagonal
                eterm = (alias[idx[2]],)
                eop = (Op(r"a^\dagger a", 0),)
            else:
                eterm = (alias[idx[2]], alias[idx[3]])
                eop = (Op(r"a^\dagger", 1), Op("a", -1))
            
    
            if idx[2] == idx[3]: # diagonal
                model[vterm+eterm].append(vop+eop+(factor,))
            else:  # off-diagonal
                model[vterm+eterm].append(vop+eop+(factor,))
                model[vterm+eterm].append(vop+eop[::-1]+(factor,))
                
        else:
            assert False
    
    order = {}
    basis = []
    if not multi_e:
        idx = 0
        for e_idx in range(2):
            order[f"e_{e_idx}"] = idx
            basis.append(ba.Basis_Simple_Electron())
            idx += 1
    else:
        order = {"e_0":0, "e_1":0}
        basis.append(ba.Basis_Multi_Electron(2,[0,0]))
        idx = 1 

    for symbol, al in alias.items():
        if al.split("_")[0] == "v":
            order[al] = int(al.split("_")[1]) + idx
            basis.append(ba.Basis_SHO(eval(f"w{symbol}"), 30))
    
    logger.info(f"order:{order}")
    logger.info(f"basis:{basis}")
    
    mol_list2 = MolList2(order, basis, model, ModelTranslator.general_model)
    mpo = Mpo(mol_list2)
    logger.info(f"mpo_bond_dims:{mpo.bond_dims}")
    mps = Mps.hartree_product_state(mol_list2,condition={"e_1":1})
    print(mps.qn, mps.qntot, mps.qnidx)
    
    compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=10)
    evolve_config = EvolveConfig(EvolveMethod.prop_and_compress)
    
    job = VibronicModelDynamics(mol_list2, mps0=mps,
                    h_mpo = mpo, 
                    compress_config=compress_config,
                    evolve_config=evolve_config)
    # warm-up
    job.evolve(evolve_dt=5.0, nsteps=3)
    
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    job.evolve_config = evolve_config
    job.latest_mps.evolve_config = evolve_config
    job.evolve(evolve_dt=80.0, nsteps=60)
    
    std = np.load("std.npz")
    assert np.allclose(std['electron occupations array'], job.e_occupations_array)
