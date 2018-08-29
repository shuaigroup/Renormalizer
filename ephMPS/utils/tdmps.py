# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

"""
todo: a wrapper for tdmps. make it a framework to deal with io and info dump and sth. like that.
resume from a pickle file
"""

import json
import os
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
        self.dump_dir = None
        self.job_name = None
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
            self.evolve_times.append(self.latest_evolve_time + evolve_dt)
            self.real_times.append(new_real_time)
            if self.dump_dir is not None and self.job_name is not None:
                self.dump_dict()
            logger.info('%s complete, time cost %s, %s' % (step_str, time_cost, new_mps))
        logger.info('%d steps of evolution with delta t = %g complete!' % (nsteps, evolve_dt))
        return self

    def evolve_single_step(self, evolve_dt):
        """
        :return: new mps after the evolution
        """
        raise NotImplementedError

    def get_dump_dict(self):
        """

        :return: return a (ordered) dict to dump as json
        """
        raise NotImplementedError

    def dump_dict(self):
        if self.dump_dir is None or self.job_name is None:
            raise ValueError('Dump dir or job name not set')
        file_path = os.path.join(self.dump_dir, self.job_name + '.json')
        bak_path = file_path + '.bak'
        if os.path.exists(file_path):
            # in case of shutdown while dumping
            if os.path.exists(bak_path):
                os.remove(bak_path)
            os.rename(file_path, bak_path)
        with open(file_path, 'w') as fout:
            json.dump(self.get_dump_dict(), fout, indent=2)
        if os.path.exists(bak_path):
            os.remove(bak_path)

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

