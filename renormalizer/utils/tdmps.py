# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>


import json
import os
import logging
from datetime import datetime

import numpy as np

# this file shouldn't import anything from the `mps` module. IOW it's mps agnostic
from renormalizer.utils.configs import EvolveConfig


logger = logging.getLogger(__name__)


class TdMpsJob(object):
    def __init__(self, evolve_config: EvolveConfig = None, dump_dir: str=None, job_name: str=None):
        logger.info(f"Creating TDMPS job. dump_dir: {dump_dir}. job_name: {job_name}")
        if evolve_config is None:
            logger.debug("using default evolve config")
            self.evolve_config: EvolveConfig = EvolveConfig()
        else:
            self.evolve_config: EvolveConfig = evolve_config
        logger.info(f"evolve_config: {self.evolve_config}")
        logger.info("Step 0/?. Preparing MPS in the initial state.")
        self.evolve_times = [0]
        # output abstract of current mps every x steps
        self.info_interval = 1
        self.dump_dir = dump_dir
        self.job_name = job_name
        mps = self.init_mps()
        if mps is None:
            raise ValueError("init_mps should return an mps. Got None")
        self.latest_mps = mps
        self.process_mps(mps)
        logger.info("TDMPS job created.")

    def init_mps(self):
        """
        :return: initial mps of the system
        """
        raise NotImplementedError

    def process_mps(self, mps):
        raise NotImplementedError

    def evolve(self, evolve_dt=None, nsteps=None, evolve_time=None):
        # deal with arguments
        if nsteps is not None and evolve_time is not None:
            logger.debug("calculate evolve_dt according to evolve_time / nsteps")
            evolve_dt = evolve_time / nsteps
                   
        if evolve_dt is None:
            # adaptive mode
            if not self.evolve_config.rk_config.adaptive:
                raise ValueError("in non-adaptive mode evolve_dt is not given")
            if evolve_time is None:
                logger.info("evolution will stop by `stop_evolve_criteria`")
                target_time = None
            else:
                target_time = self.evolve_times[-1] + evolve_time
            target_steps = "?"
            nsteps = int(1e10)  # insanely large
        else:
            if nsteps is None and evolve_time is None:
                raise ValueError(
                    "Must provide number of steps or evolve time"
                )
            if evolve_time is None:
                evolve_time = nsteps * evolve_dt
            elif nsteps is None:
                nsteps = int(evolve_time // evolve_dt)
            else:
                logger.warning(
                    "Both nsteps and evolve_time is defined for evolution. The result may be unexpected."
                )
            target_steps = len(self.evolve_times) + nsteps - 1
            target_time = self.evolve_times[-1] + evolve_time

        wall_times = [datetime.now()]
        if self.evolve_config.adaptive and evolve_dt is not None:
            logger.warning("evolve_dt is ignored in adaptive propagation")
        for i in range(nsteps):
            if target_time is not None and abs(self.latest_evolve_time - target_time) < 1e-3:
                break
            if self.evolve_config.adaptive:
                evolve_dt = self.evolve_config.evolve_dt
                assert not (np.iscomplex(evolve_dt) ^ np.iscomplex(target_time))
            new_evolve_time = self.latest_evolve_time + evolve_dt
            step_str = "step {}/{}, time {}/{}".format(
                len(self.evolve_times), target_steps, new_evolve_time, target_time
            )
            self.evolve_times.append(new_evolve_time)
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
            self.latest_mps = new_mps
            self.process_mps(new_mps)
            new_wall_time = datetime.now()
            time_cost = new_wall_time - wall_times[-1]
            if self.info_interval is not None and i % self.info_interval == 0:
                mps_abstract = str(new_mps)
            else:
                mps_abstract = ""
            logger.info(
                "%s complete, time cost %s. %s" % (step_str, time_cost, mps_abstract)
            )
            wall_times.append(new_wall_time)
            if self._defined_output_path:
                try:
                    self.dump_dict()
                except IOError:  # never quit calculation because of IOError
                    logger.exception("dumping dict failed with IOError")
            if self.stop_evolve_criteria():
                logger.info(
                    "Criteria to stop the evolution has met. Stop the evolution"
                )
                break

        logger.info(f"{len(wall_times)-1} steps of evolution complete!")
        logger.info(
            "Normal termination. Time cost: %s" % (wall_times[-1] - wall_times[0])
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
        os.makedirs(self.dump_dir, exist_ok=True)
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
    def latest_evolve_time(self):
        return self.evolve_times[-1]

    @property
    def evolve_times_array(self):
        return np.array(self.evolve_times)

    @property
    def _defined_output_path(self):
        return self.dump_dir is not None and self.job_name is not None
