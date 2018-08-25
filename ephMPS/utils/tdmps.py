# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

"""
todo: a wrapper for tdmps. make it a framework to deal with io and info dump and sth. like that.
resume from a pickle file
"""

import logging
from datetime import datetime

import numpy as np


logger = logging.getLogger(__name__)


class TdMpsJob(object):

    def __init__(self):
        init_mps = self.init_mps()
        self.evolve_times = [0]
        self.tdmps_list = [init_mps]
        self.real_times = [datetime.now()]

    def init_mps(self):
        """
        :return: initial mps of the system
        """
        raise NotImplementedError

    def evolve(self, evolve_dt, nsteps):
        for i in range(nsteps - 1):
            new_mps = self.evolve_single_step(evolve_dt=evolve_dt)
            new_real_time = datetime.now()
            logger.info('step %d, time cost %s, %s' % (i, new_real_time - self.latest_real_time, new_mps))
            self.tdmps_list.append(new_mps)
            self.evolve_times.append(self.latest_evolve_time + evolve_dt)
            self.real_times.append(new_real_time)
        return self

    def evolve_single_step(self, evolve_dt):
        """
        :return: new mps after the evolution
        """
        raise NotImplementedError

    @property
    def latest_mps(self):
        return self.tdmps_list[-1]

    @property
    def latest_evolve_time(self):
        return self.evolve_times[-1]

    @property
    def latest_real_time(self):
        return self.real_times[-1]

