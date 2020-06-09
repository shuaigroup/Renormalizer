# -*- coding: utf-8 -*-
# Author: Tong Jiang <tongjiang1000@gmail.com>
# correction vector base
import numpy as np
from multiprocessing import Pool
import multiprocessing
from renormalizer.mps import Mpo
from renormalizer.utils import Quantity
from renormalizer.utils.elementop import construct_e_op_dict, ph_op_matrix
import importlib.util
import logging
from typing import List

logger = logging.getLogger(__name__)

def batch_run(freq_reg, cores, obj):
    """
    batch run of cv calculation
    freq_reg: list object, frequecny windown
    cores: number of cores to be used in multiprocessing calculation
    obj: SpectraZtCV or SpectraFtCV
    """
    logger.info(f"{len(freq_reg)} total frequency points to do")
    spectra = []

    if cores > 1:
        # multiprocessing
        if importlib.util.find_spec("cupy"):
            multiprocessing.set_start_method('forkserver', force=True)
        pool = Pool(processes=cores)
        logger.info(f"{cores} multiprocess parallelization activated")
        for i_spec in pool.imap(obj.cv_solve, freq_reg):
            spectra.append(i_spec)

    elif cores == 1:
        # single process
        for omega in freq_reg:
            spectra.append(obj.cv_solve(omega))
            #obj.cv_mps.ensure_left_canon()
            obj.clear_res()
    else:
        assert False

    return spectra


class SpectraCv(object):
    def __init__(
        self,
        mol_list,
        spectratype,
        m_max,
        eta,
        h_mpo = None,  
        method = "1site",
        procedure_cv = None,
        rtol = 1e-5,
        b_mps = None,
        e0 = None,
        cv_mps = None,
    ):
        
        self.mol_list = mol_list
        
        assert spectratype in ["abs", "emi", None]
        self.spectratype = spectratype
        
        self.m_max = m_max
        self.eta = eta
        
        # Hamiltonian
        if h_mpo is None:
            self.h_mpo = Mpo(mol_list)
        else:
            self.h_mpo = h_mpo

        assert method in ["1site", "2site"]
        self.method = method
        logger.info(f"cv optimize method: {method}")
        
        # percent used to update correction vector for each isweep process
        # see function mps.lib.select_basis
        if procedure_cv is None:
            procedure_cv = [0.4, 0.4, 0.2, 0.2, 0.1, 0.1] + [0] * 45
        self.procedure_cv = procedure_cv
        self.rtol = rtol
        
        # ax=b b_mps and ground state energy e0
        if b_mps is None:
            self.b_mps, self.e0 = self.init_b_mps()
        else:
            self.b_mps = b_mps
            # e0 is used in zero temperature case
            self.e0 = e0
        
        # initial_guess cv_mps
        if cv_mps is None:
            self.cv_mps = self.init_cv_mps()
        else:
            self.cv_mps = cv_mps
        
        # results
        self.hop_time = []
        self.macro_iteration_result = []

        logger.info("DDMRG job created.")

    #def check_qn(self, X):
    #    check_op = Mpo.onsite(self.mol_list, r"a^\dagger a", dipole=False)
    #    check_1 = X.conj().dot(check_op.apply(X)) / X.conj().dot(X)
    #    print('Quantum number of X', check_1)
    
    def cv_solve(self, omega):
        
        converged = False
        isweep = 1
        len_cv = len(self.cv_mps)
        self.oper_prepare(omega)
        
        while isweep < len(self.procedure_cv):
            if isweep % 2 == 0:
                direction = 'right'
                if self.method == '1site':
                    irange = np.arange(1, len_cv+1)
                else:
                    irange = np.arange(2, len_cv+1)
            else:
                direction = 'left'
                if self.method == '1site':
                    irange = np.arange(len_cv, 0, -1)
                else:
                    irange = np.arange(len_cv, 1, -1)
            if isweep == 1:
                lr_group = self.initialize_LR(direction)
            
            micro_iteration_result = []
            for isite in irange:
                l_value = self.optimize_cv(
                    lr_group, direction,
                    isite, isweep, percent=self.procedure_cv[isweep-1])
                if (self.method == '1site') & (
                    ((direction == 'left') & (isite == 1)) or (
                        (direction == 'right') & (isite == len_cv))):
                    pass
                else:
                    lr_group = \
                        self.update_LR(lr_group, direction, isite)
                micro_iteration_result.append(-1./(np.pi * self.eta)*l_value)
                logger.info(f"cv_bond_dims: {self.cv_mps.bond_dims}")
                logger.debug(f"omega:{omega}, isweep:{isweep}, isite:{isite}, response result:{micro_iteration_result[-1]}")
            
            self.macro_iteration_result.append(max(micro_iteration_result))
            isweep += 1
            
            # breaking condition, depending on problem,
            # can make it more strict
            # by requiring the minimum isweep number as well as the tol
            if (isweep > 1) and self.procedure_cv[isweep-1] == 0:
                v1, v2 = sorted(self.macro_iteration_result)[-2:]
                if abs((v1-v2)/v1) < self.rtol:
                    converged = True
                    break
        if converged:
            logger.info("cv converged!")
        else:
            logger.warning("cv *NOT* converged!")

        logger.info(f"omega:{omega}, sweeps:{isweep}, average_hop:{int(np.mean(self.hop_time))},res:{max(self.macro_iteration_result)}")
        
        return max(self.macro_iteration_result)
    
    def clear_res(self):
        # reuse the same object to calculate another frequency
        self.hop_time.clear()
        self.macro_iteration_result.clear()

    def init_cv_mps(self):
        raise NotImplementedError

    def init_b_mps(self):
        raise NotImplementedError
    
    def oper_prepare(self, omgea):
        raise NotImplementedError

    def optimize_cv(self, lr_group, direction, isite, isweep, percent=0):
        raise NotImplementedError

    def initialize_LR(self, direction):
        raise NotImplementedError

    def update_LR(self, lrgroup, direction, isite):
        raise NotImplementedError
