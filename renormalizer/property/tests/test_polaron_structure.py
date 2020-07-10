'''
The 1-d Holstein model of qiangshi's paper JCP 142, 174103 (2015)
molecular distance : 5 A
omega              : 40 cm^-1
mass               : 250 amu
e-ph coupling      : 3500 cm^-1 / A
J                  : 50, 100, 200, 400 cm^-1
nsites             : 20 

map to our MPO
the unit of coupling constant c in our code has unit energy / mass^{1/2} * length

3500 cm^-1 / (1*agstrom2au) mass^(1/2))
'''
from renormalizer.utils import constant
from renormalizer.model import  Phonon, Mol, MolList
from renormalizer.mps import Mpo, MpDm, ThermalProp, gs
from renormalizer.utils import Quantity, EvolveConfig, EvolveMethod

from renormalizer.property import Property, ops
from renormalizer.utils import log

import numpy as np
import pytest
import logging

logger = logging.getLogger(__name__)
# for debug
dump_dir = None
job_name = None
#dump_dir = "./"
#job_name = "J400"
#log.register_file_output(dump_dir+job_name+".log", mode="w")

# set the model
omega_value = 40. * constant.cm2au
c_value = 3500. * constant.cm2au / (250.*constant.amu2au)**0.5 / constant.angstrom2au 
D_value = c_value / omega_value**2
g_value = np.sqrt(omega_value / 2) * D_value
lambda_value = g_value**2 * omega_value

nmols = 5
dipole_abs = 1.
elocalex = 0. 
j_value = 400 * constant.cm2au

logger.info(f"g:{g_value}") 
logger.info(f"lambda:{lambda_value*constant.au2ev}ev,{lambda_value*constant.au2cm}cm^-1") 
logger.info(f"J:{j_value*constant.au2ev}ev, {j_value*constant.au2cm}cm^-1")

j_matrix = np.diag(np.ones(nmols-1)*j_value,k=-1)
j_matrix += j_matrix.T

ph_phys_dim = 5

omega = [Quantity(omega_value),Quantity(omega_value)]
D = [Quantity(0.),Quantity(D_value)]

ph = Phonon(omega, D, ph_phys_dim)

mol_list = MolList([Mol(Quantity(elocalex), [ph], dipole_abs)] * nmols,
        j_matrix, scheme=3)

# periodic nearest-neighbour interaction
mpo = Mpo(mol_list)
periodic = Mpo.intersite(mol_list, {0: r"a^\dagger", nmols - 1: "a"}, {},
                         Quantity(j_value))
mpo = mpo.add(periodic).add(periodic.conj_trans())


@pytest.mark.parametrize("periodic",(True, False))
def test_thermal_equilibrium(periodic):

    if periodic:
        # define properties
        # periodic case
        prop_mpos = ops.e_ph_static_correlation(mol_list, periodic=True)
        prop_strs = list(prop_mpos.keys())
        prop_strs.append("e_rdm")
        prop = Property(prop_strs, prop_mpos)
    else:
        # non-periodic case (though actually periodic)
        prop_mpos = {}
        for imol in range(nmols):
            prop_mpo = ops.e_ph_static_correlation(mol_list, imol=imol)
            prop_mpos.update(prop_mpo)
        prop_strs = list(prop_mpos.keys())
        prop_strs.append("e_rdm")
        prop = Property(prop_strs, prop_mpos)
    
    beta = Quantity(1500., "K").to_beta()
    logger.info(f"beta:{beta}")
    
    nsteps = 1
    dbeta = beta / nsteps / 2j
    
    evolve_config = EvolveConfig(method=EvolveMethod.prop_and_compress, adaptive=True,
            adaptive_rtol=1e-4, guess_dt=0.1/1j)
    
    init_mpdm = MpDm.max_entangled_ex(mol_list)
    #init_mpdm.compress_config.bond_dim_max_value=10
    init_mpdm.compress_config.threshold = 1e-4
    
    td = ThermalProp(init_mpdm, mpo, evolve_config=evolve_config,
            dump_dir=dump_dir, job_name=job_name, properties=prop)
    td.evolve(dbeta, nsteps=nsteps)
    
    if periodic:
        def combine(local_prop):
            res = []
            for dis in range(nmols):
                res.append(local_prop.prop_res["S_"+str(dis)+"_0"][-1])
            return res
    else:
        def combine(local_prop):
            e_ph_static_corr = []
            for dis in range(nmols):
                res = 0.
                for i in range(nmols):
                    res = res + np.array(local_prop.prop_res["S_"+str(i)+"_"+str((i+dis)%nmols)+"_0"][-1])
                e_ph_static_corr.append(res)
            return  e_ph_static_corr

    assert np.allclose(td.properties.prop_res["e_rdm"][-1], thermal_std["e_rdm"])
    assert np.allclose(combine(td.properties), thermal_std["e_ph_static_corr"])
	
    # directly calculate properties
    mpdm = td.latest_mps
    prop.calc_properties(mpdm, None)
    assert np.allclose(prop.prop_res["e_rdm"][-1], prop.prop_res["e_rdm"][-2])

thermal_std = {
"e_ph_static_corr": [
      0.07140736648696919,
      0.001362988559264886,
      4.475610540677275e-05,
      3.0216826954447922e-05,
      0.001325628780368527
    ],
"e_rdm": [
      [
        0.19191446386636984,
        -0.07099664999188746,
        0.013367027706242202,
        -0.0017135694987144776,
        0.0001689278296078954
      ],
      [
        -0.07099664999188746,
        0.20533935787738303,
        -0.07268795998295394,
        0.013529497347607916,
        -0.0017135733884236562
      ],
      [
        0.013367027706242207,
        -0.07268795998295394,
        0.20549318292347962,
        -0.07268883054416064,
        0.013366869686751964
      ],
      [
        -0.0017135694987144763,
        0.013529497347607921,
        -0.07268883054416062,
        0.20534869484572402,
        -0.07099570692510888
      ],
      [
        0.00016892782960789535,
        -0.001713573388423658,
        0.013366869686751964,
        -0.0709957069251089,
        0.1919043004870439
      ]
    ]
}
