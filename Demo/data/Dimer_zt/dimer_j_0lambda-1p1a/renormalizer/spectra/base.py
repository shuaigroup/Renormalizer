# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

import numpy as np

from renormalizer.mps import Mpo
from renormalizer.utils import TdMpsJob, Quantity


class SpectraTdMpsJobBase(TdMpsJob):
    def __init__(
        self,
        model,
        spectratype,
        temperature,
        evolve_config=None,
        offset=Quantity(0),
        dump_dir=None,
        job_name=None
    ):
        self.model = model
        assert spectratype in ["emi", "abs"]
        self.spectratype = spectratype
        if spectratype == "emi":
            self.nexciton = 1
        else:
            self.nexciton = 0
        self.temperature = temperature
        self.h_mpo: Mpo = Mpo(model, offset=offset)
        self._autocorr = []
        super(SpectraTdMpsJobBase, self).__init__(evolve_config=evolve_config, dump_dir=dump_dir, job_name=job_name)

    def process_mps(self, braket_pair):
        self._autocorr.append(braket_pair.ft)

    @property
    def autocorr(self):
        return np.array(self._autocorr)
