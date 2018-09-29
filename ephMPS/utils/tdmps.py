# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

"""
todo: a wrapper for tdmps. make it a framework to deal with io and info dump and sth. like that.
resume from a pickle file
"""

import json
import os
import errno
import logging
from datetime import datetime, timedelta

import numpy as np


logger = logging.getLogger(__name__)


def predict_time(real_times, nsteps):
    time_deltas = [(time - real_times[0]).total_seconds() for time in real_times]
    current_steps = len(real_times)
    p_c = np.polyfit(range(current_steps), time_deltas, 2)
    # can only predict 2 * current_steps accurately. Further prediction is not possible
    target_step = min(2 * current_steps, nsteps)
    target_total_seconds = float(np.polyval(p_c, target_step))
    target_time = real_times[0] + timedelta(seconds=target_total_seconds)
    return target_step, target_time


class TdMpsJob(object):

    def __init__(self):
        logger.info('Creating TDMPS job.')
        logger.info('Step 1/?. Preparing MPS in the intial state.')
        init_mps = self.init_mps()
        self.evolve_times = [0]
        self.tdmps_list = [init_mps]
        self.dump_dir = None
        self.job_name = None
        logger.info('TDMPS job created.')

    def init_mps(self):
        """
        :return: initial mps of the system
        """
        raise NotImplementedError

    def _evolve(self, evolve_dt, nsteps):
        target_steps = len(self.tdmps_list) + nsteps - 1
        real_times = [datetime.now()]
        for i in range(nsteps - 1):
            step_str = 'step %d/%d' % (len(self.tdmps_list) + 1, target_steps)
            logger.info('%s begin.' % step_str)
            new_mps = self.evolve_single_step(evolve_dt=evolve_dt)
            new_real_time = datetime.now()
            time_cost = new_real_time - real_times[-1]
            self.tdmps_list.append(new_mps)
            self.evolve_times.append(self.latest_evolve_time + evolve_dt)
            logger.info('%s complete, time cost %s, evolve time %g. %s'
                        % (step_str, time_cost, self.latest_evolve_time, new_mps))
            real_times.append(new_real_time)
            if 10 < i:  # otherwise samples too small to make a prediction
                predict_step, predicted_time = predict_time(real_times, nsteps)
                logger.info('predict %s at step %d.' % (predicted_time, predict_step))
            if self.dump_dir is not None and self.job_name is not None:
                self.dump_dict()
            if self.stop_evolve_criteria():
                logger.info('Criteria to stop the evolution has met. Stop the evolution')
                break
        logger.info('%d steps of evolution with delta t = %g complete!' % (nsteps, evolve_dt))
        logger.info('Time cost: %s' % (real_times[-1] - real_times[0]))

    def evolve(self, evolve_dt, nsteps):
        try:
            self._evolve(evolve_dt, nsteps)
        except:
            # todo: find common exceptions and state explicitly here
            logger.exception('An exception occurs during the evolution')
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
        # todo: refactor with `pathlib` which is not compatible with python2 (maybe after year 2020!)
        try:
            os.makedirs(self.dump_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
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

    def set_threshold(self, threshold):
        logger.info('Set threshold to %g' % threshold)
        self.latest_mps.threshold = threshold

    def stop_evolve_criteria(self):
        return False

    @property
    def latest_mps(self):
        return self.tdmps_list[-1]

    @property
    def latest_evolve_time(self):
        return self.evolve_times[-1]

    @property
    def evolve_times_array(self):
        return np.array(self.evolve_times)
