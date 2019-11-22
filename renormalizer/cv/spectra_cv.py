# -*- coding: utf-8 -*-
# Author: Tong Jiang <tongjiang1000@gmail.com>
# correction vector base
import numpy as np
from multiprocessing import Pool
from renormalizer.mps import Mpo
from renormalizer.utils import Quantity
from renormalizer.utils.elementop import construct_e_op_dict, ph_op_matrix
import logging
import time

logger = logging.getLogger(__name__)


def unwrap_self(arg, **kwarg):
    return SpectraCv.cv_solve(*arg, **kwarg)


class SpectraCv(object):
    def __init__(
        self,
        mol_list,
        spectratype,
        freq_reg,
        m_max,
        eta,
        method="1site",
        procedure_cv=None,
        cores=1
    ):
        self.freq_reg = freq_reg
        self.m_max = m_max
        self.eta = eta
        self.mol_list = mol_list
        assert spectratype in ["abs", "emi"]
        self.spectratype = spectratype
        # percent used to update correction vector for each sweep process
        # see function mps.lib.select_basis
        if procedure_cv is None:
            procedure_cv = [0.4, 0.4, 0.2, 0.2, 0.1, 0.1] + [0] * 45
        self.procedure_cv = procedure_cv
        self.method = method
        self.cores = cores
    '''
    def check_qn(self, X):
        check_op = Mpo.onsite(self.mol_list, r"a^\dagger a", dipole=False)
        check_1 = X.conj().dot(check_op.apply(X)) / X.conj().dot(X)
        print('Quantum number of X', check_1)
    '''
    def run(self):
        start = time.time()
        pool = Pool(processes=self.cores)
        logger.info(f"{len(self.freq_reg)} total frequency to do")
        logger.info(f"{self.cores} multiprocess parallelization activated")
        freq_reg = self.freq_reg
        spectra = []
        for i_spec in pool.imap(
                unwrap_self, zip([self]*len(freq_reg), freq_reg)
        ):
            spectra.append(i_spec)
        logger.info(f"time used:{time.time()-start}")
        return spectra

    def cv_solve(self, omega):
        self.hop_time = []
        logger.info(f'begin calculate omega:{omega}')
        # progress = (omega-self.freq_reg[0])/(
        # self.freq_reg[-1]-self.freq_reg[0])*100
        # logger.info(f"procesees:{progress}/{100}")

        total_num = 1
        spectra = []

        # for omega in freq region:
        for omega in [omega]:
            num = 1
            result = []
            len_cv = len(self.cv_mps)
            self.oper_prepare(omega)

            while num < len(self.procedure_cv):
                # use the optimized cv_mps of previous omega as guess
                if total_num % 2 == 0:
                    direction = 'right'
                    if self.method == '1site':
                        irange = np.array(range(1, len_cv+1))
                    else:
                        irange = np.array(range(2, len_cv+1))
                else:
                    direction = 'left'
                    if self.method == '1site':
                        irange = np.array(range(len_cv, 0, -1))
                    else:
                        irange = np.array(range(len_cv, 1, -1))
                if num == 1:
                    lr_group = self.initialize_LR(direction)
                for isite in irange:
                    l_value = self.optimize_cv(
                        lr_group, direction,
                        isite, num, percent=self.procedure_cv[num-1])
                    if (self.method == '1site') & (
                        ((direction == 'left') & (isite == 1)) or (
                            (direction == 'right') & (isite == len_cv))):
                        pass
                    else:
                        lr_group = \
                            self.update_LR(lr_group, direction, isite)
                result.append(l_value)
                num += 1
                total_num += 1
                # breaking condition, depending on problem,
                # can make it more strict
                # by requiring the minimum sweep number as well as the tol
                if num > 3:
                    if abs((result[-1] - result[-3]) / result[-1]) < 0.001:
                        break
            spectra.append((-1./(np.pi * self.eta)) * result[-1])
        logger.info(f"omega:{omega}, sweep:{num},average_hop:{int(np.mean(self.hop_time))}")
        logger.info(f'omega:{omega} calculation complete')
        return spectra

    def init_mps(self):
        raise NotImplementedError

    def init_oper(self):
        raise NotImplementedError

    def oper_prepare(self, omega):
        raise NotImplementedError

    def initialize_LR(self, direction):
        raise NotImplemented

    def update_LR(self, direction):
        raise NotImplemented
