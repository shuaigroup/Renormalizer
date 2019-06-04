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

# this file shouldn't import anythin from the `mps` module. IOW it's mps agnostic
from ephMPS.utils.configs import EvolveConfig

logger = logging.getLogger(__name__)

# this is now highly inaccurate. Could use some more powerful algorithms (see `fbprophet`)
# but is it worth the fuss?
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
    def __init__(self, evolve_config: EvolveConfig = None, dump_dir: str=None, job_name: str=None):
        logger.info("Creating TDMPS job.")
        if evolve_config is None:
            logger.debug("using default evolve config")
            self.evolve_config: EvolveConfig = EvolveConfig()
        else:
            self.evolve_config: EvolveConfig = evolve_config
        logger.info("Step 0/?. Preparing MPS in the initial state.")
        self.evolve_times = [0]
        # output abstract of current mps every x steps
        self.info_interval = 1
        self.dump_dir = dump_dir
        self.job_name = job_name
        self.tdmps_list = [self.init_mps()]
        logger.info("TDMPS job created.")

    def init_mps(self):
        """
        :return: initial mps of the system
        """
        raise NotImplementedError

    def evolve(self, evolve_dt=None, nsteps=None, evolve_time=None):
        # deal with arguments
        if nsteps is not None and evolve_time is not None:
            logger.debug("calculate evolve_dt according to the rest two args")
            evolve_dt = evolve_time / nsteps
        if evolve_dt is None:
            # adaptive mode
            if not self.evolve_config.rk_config.adaptive:
                raise ValueError("in non-adaptive mode evolve_dt is not given")
            if evolve_time is None:
                target_time = None  # stop by `stop_evolve_criteria`
            else:
                target_time = self.evolve_times[-1] + evolve_time
            target_steps = "?"
            nsteps = int(1e10)  # insanely large
        else:
            if nsteps is None:
                raise ValueError(
                    "Must provide number of steps"
                )
            if evolve_time is None:
                evolve_time = nsteps * evolve_dt
            else:
                logger.warning(
                    "Both nsteps and evolve_time is defined for evolution. The result may be unexpected."
                )
            target_steps = len(self.tdmps_list) + nsteps - 1
            target_time = self.evolve_times[-1] + evolve_time

        real_times = [datetime.now()]
        if self.evolve_config.adaptive and evolve_dt is not None:
            logger.warning("evolve_dt is ignored in adaptive propagation")
        for i in range(nsteps):
            if target_time is not None and abs(self.latest_evolve_time - target_time) < 1e-3:
                break
            if self.evolve_config.adaptive:
                evolve_dt = self.evolve_config.evolve_dt
                assert not (np.iscomplex(evolve_dt) ^ np.iscomplex(target_time))
            new_evolve_time = self.latest_evolve_time + evolve_dt
            self.evolve_times.append(new_evolve_time)
            step_str = "step {}/{}, time {}/{}".format(
                len(self.tdmps_list), target_steps, new_evolve_time, target_time
            )
            logger.info("{} begin.".format(step_str))
            # XXX: the actual evolve step here
            new_mps = self.evolve_single_step(evolve_dt=evolve_dt)
            if self.evolve_config.adaptive:
                # update evolve_dt
                if target_time is not None \
                        and abs(target_time) < abs(new_evolve_time) + abs(new_mps.evolve_config.evolve_dt):
                    new_dt = target_time - new_evolve_time
                    new_mps.evolve_config.evolve_dt = new_dt
                else:
                    new_dt = new_mps.evolve_config.evolve_dt
                self.evolve_config.evolve_dt = new_dt
            new_real_time = datetime.now()
            time_cost = new_real_time - real_times[-1]
            if hasattr(self.tdmps_list[-1], "clear_memory"):
                self.tdmps_list[-1].clear_memory()
            else:
                logger.debug(f"No clear memory method for type {type(self.tdmps_list[-1])} found")
            self.tdmps_list.append(new_mps)
            if self.info_interval is not None and i % self.info_interval == 0:
                mps_abstract = str(new_mps)
            else:
                mps_abstract = ""
            logger.info(
                "%s complete, time cost %s. %s" % (step_str, time_cost, mps_abstract)
            )
            real_times.append(new_real_time)
            """
            if 10 < i:  # otherwise samples too small to make a prediction
                predict_step, predicted_time = predict_time(real_times, nsteps)
                logger.info("predict %s at step %d." % (predicted_time, predict_step))
            """
            if self._defined_output_path:
                try:
                    self.dump_dict()
                except IOError as e:  # never quit calculation because of IOError
                    logger.warning("dumping dict failed with IOError")
                    logger.warning(e)
            if self.stop_evolve_criteria():
                logger.info(
                    "Criteria to stop the evolution has met. Stop the evolution"
                )
                break

        logger.info(f"{len(real_times)-1} steps of evolution complete!")
        logger.info(
            "Normal termination. Time cost: %s" % (real_times[-1] - real_times[0])
        )
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
        if not self._defined_output_path:
            raise ValueError("Dump dir or job name not set")
        # todo: refactor with `pathlib`
        try:
            os.makedirs(self.dump_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        file_path = os.path.join(self.dump_dir, self.job_name + ".json")
        bak_path = file_path + ".bak"
        if os.path.exists(file_path):
            # in case of shutdown while dumping
            if os.path.exists(bak_path):
                os.remove(bak_path)
            os.rename(file_path, bak_path)
        d = self.get_dump_dict()
        with open(file_path, "w") as fout:
            json.dump(d, fout, indent=2)
        if os.path.exists(bak_path):
            os.remove(bak_path)

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

    @property
    def _defined_output_path(self):
        return self.dump_dir is not None and self.job_name is not None