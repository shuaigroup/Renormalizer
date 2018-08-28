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
try:
    import seaborn as sns
except ImportError:
    logging.warn('Seaborn not installed, draw module down')
    sns = None
try:
    from matplotlib import pyplot as plt
except ImportError:
    logging.warn('Matplotlib not installed, draw module down')
    plt = None


logger = logging.getLogger(__name__)


class TdMpsJob(object):

    def __init__(self):
        logger.info('Creating TDMPS job.')
        logger.info('Step 1/?. Preparing MPS in the intial state.')
        init_mps = self.init_mps()
        self.evolve_times = [0]
        self.tdmps_list = [init_mps]
        self.real_times = [datetime.now()]
        logger.info('TDMPS job created.')

    def init_mps(self):
        """
        :return: initial mps of the system
        """
        raise NotImplementedError

    def evolve(self, evolve_dt, nsteps):
        target_steps = len(self.tdmps_list) + nsteps - 1
        for i in range(nsteps - 1):
            step_str = 'step %d/%d' % (len(self.tdmps_list) + 1, target_steps)
            logger.info('%s begin' % step_str)
            new_mps = self.evolve_single_step(evolve_dt=evolve_dt)
            new_real_time = datetime.now()
            time_cost = new_real_time - self.latest_real_time
            self.tdmps_list.append(new_mps)
            logger.info('%s complete, time cost %s, %s' % (step_str, time_cost, new_mps))
            self.evolve_times.append(self.latest_evolve_time + evolve_dt)
            self.real_times.append(new_real_time)
        logger.info('%d steps of evolution with delta t = %g complete!' % (nsteps, evolve_dt))
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

    @property
    def evolve_times_array(self):
        return np.array(self.evolve_times)

