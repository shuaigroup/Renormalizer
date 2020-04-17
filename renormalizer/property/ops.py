from renormalizer.mps import Mpo
from renormalizer.utils import Quantity, Op
from renormalizer.model import MolList, MolList2, ModelTranslator
import numpy as np


def e_ph_static_correlation(mol_list: MolList, imol:int =0, jph:int =0,
        periodic:bool =False, name:str="S"):
    '''
    construct the electron-phonon static correlation operator in polaron problem
    The details of the definition, see 
    Qiang Shi et al. J. Chem. Phys. 142, 174103 (2015) or
    Romero et al. Journal of Luminescence 83-84 (1999) 147-153

    if periodic:
        # D is the displacement between different PES of each mode
        S_(m, jph) = \frac{1}{D_m+n,jph} \sum_n \langle x_{m+n,jph} a_n^\dagger a_n \rangle
        operator name = "_".join([name, str(m), str(jph)])
        m stands for distance if periodic
    else:
        S_(n,m,jph) = \frac{1}{D_m,jph} \langle x_{m, jph} a_n^\dagger a_n \rangle
        operator name: "_".join([name, str(n), str(m), str(jph)]) 
    
    Parameters:
        mol_list : MolList
            the molecular information
        imol : int
            electron site index (default:0)
            if periodic is True, imol is omitted.
        jph  : int 
            phonon site index
        periodic :  bool
            if homogenous periodic system
        name: str
            the name of the operator
    
    Note: Only one mode Holstein Model has been tested
    '''
    
    if mol_list.scheme == 4:
        raise NotImplementedError
    
    prop_mpos = {}
    
    nmols = mol_list.mol_num
    
    if not periodic:
        # each jmol site is calculated separately
        for jmol in range(nmols):
            op_name = "_".join([name, str(imol), str(jmol), str(jph)])
            prop_mpos[op_name] = Mpo.intersite(mol_list, {imol:r"a^\dagger a"}, {(jmol,
                jph):r"b^\dagger + b"},
                scale=Quantity(np.sqrt(1./2.0/mol_list[jmol].dmrg_phs[jph].omega[0])/mol_list[jmol].dmrg_phs[jph].dis[1]))
        # normalized by the displacement D
    else:
        # each distance is calculated seperately
        for dis in range(nmols):
            dis_list = []
            for jmol in range(nmols):
                kmol = (jmol+dis) % nmols
                dis_list.append(Mpo.intersite(mol_list, {jmol:r"a^\dagger a"},
                    {(kmol, jph):r"b^\dagger + b"},
                    scale=Quantity(np.sqrt(1./2.0/mol_list[kmol].dmrg_phs[jph].omega[0])/mol_list[kmol].dmrg_phs[jph].dis[1])))
            for item in dis_list[1:]:
                dis_list[0] = dis_list[0].add(item)
            op_name = "_".join([name, str(dis), str(jph)])
            prop_mpos[op_name] = dis_list[0]

    return prop_mpos

def x_average(mol_list: MolList2):
    """
    <x> of vibrational DoF
    """
    assert isinstance(mol_list, MolList2)

    mpos = []
    for v_dof in mol_list.v_dofs:
        model = {(v_dof,):[(Op("x",0),1.0)]}
        mpo = Mpo.general_mpo(mol_list, model=model,
                model_translator=ModelTranslator.general_model)
        mpos.append(mpo)

    return {"x": mpos}

def x_square_average(mol_list: MolList2):
    """
    <x^2> of vibrational DoF
    """
    assert isinstance(mol_list, MolList2)

    mpos = []
    for v_dof in mol_list.v_dofs:
        model = {(v_dof,):[(Op("x^2",0),1.0)]}
        mpo = Mpo.general_mpo(mol_list, model=model,
                model_translator=ModelTranslator.general_model)
        mpos.append(mpo)

    return {r"x^2": mpos}
