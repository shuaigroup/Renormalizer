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
    def __init__(self, evolve_config: EvolveConfig = None, dump_mps: str=None, dump_dir: str=None, job_name: str=None):
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
        # dump mps, None: not dumped, "all": dump all according to interval,  
        # "one": dump only the latest mps according to interval
        if dump_mps in [None, "all", "one"]:
            self.dump_mps = dump_mps
        else:
            raise ValueError(f"dump_mps should be None, 'all', 'one'. Got {dump_mps}")

        self._dump_mps = None
        self.dump_dir = dump_dir
        self.job_name = job_name
        mps = self.init_mps()
        logger.info(f"Initial MPS: {str(mps)}")
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
        """
        Process the newly evolved the mps. Primarily for the calculation of properties.
        Note that currently ``self.latest_mps`` has not been updated.

        Parameters
        ==========
        mps :
             The evolved new mps of the time step.
        """
        raise NotImplementedError
    
    def evolve(self, evolve_dt=None, nsteps=None, evolve_time=None):
        '''
        Args:
            evolve_dt (float): the time step to run `evolve_single_step` and `process_mps`.
            nsteps (int): the total number of evolution steps
            evolve_time (float): the total evolution time
        
        .. note::
            ``evolve_dt`` math: `\times` ``nsteps`` = ``evolve_time``,  otherwise nsteps has a higher priority.
        '''
        # deal with arguments
        if (evolve_dt is not None) and  (nsteps is not None) and (evolve_time is not None):
            logger.warning("Both evolve_time and nsteps are defined for evolution. The evolve_time is omitted")
            case = 1

        elif (evolve_dt is None) and  (nsteps is not None) and (evolve_time is not None):
            evolve_dt = evolve_time / float(nsteps)
            logger.info(f"The evolve_dt is {evolve_dt}")
            case = 1

        elif (evolve_dt is not None) and  (nsteps is not None) and (evolve_time is None):
            case = 1
            
        elif (evolve_dt is not None) and  (nsteps is None) and (evolve_time is not None):
            logger.debug("calculate nsteps according to evolve_time / evolve_dt")
            nsteps = int(abs(evolve_time) // abs(evolve_dt)) + 1
            case = 1

        elif (evolve_dt is not None) and  (nsteps is None) and (evolve_time is None):
            logger.info("evolution will stop by `stop_evolve_criteria`")
            nsteps = int(1e10)  # insanely large
            case = 2
        else:
            raise ValueError(f"The input parameters evolve_dt:{evolve_dt}, nsteps:{nsteps}, evolve_time:{evolve_time} do not meet the requirements!")

        if case == 1:
            # evolution controlled by evolve_dt and nsteps
            target_steps = len(self.evolve_times) + nsteps - 1
            target_time = self.evolve_times[-1] + nsteps * evolve_dt
        elif case == 2:
            # evolution controlled by `stop_evolve_criteria`
            target_steps = "?"
            target_time = "?"

        wall_times = [datetime.now()]

        for i in range(nsteps):

            if self.stop_evolve_criteria():
                logger.info(
                    "Criteria to stop the evolution has met. Stop the evolution"
                )
                break

            step_str = "step {}/{}, at time {}/{}".format(
                len(self.evolve_times), target_steps, self.latest_evolve_time, target_time
            )
            logger.info("{} begin.".format(step_str))
            
            # evolve
            new_mps = self.evolve_single_step(evolve_dt)
            
            # process
            self.evolve_times.append(self.latest_evolve_time + evolve_dt)
            self.process_mps(new_mps)
            self.latest_mps = new_mps
            
            # wall time
            evolution_wall_time = datetime.now()
            time_cost = evolution_wall_time - wall_times[-1]
            wall_times.append(evolution_wall_time)
            
            # output information
            if self.info_interval is not None and i % self.info_interval == 0:
                mps_abstract = str(new_mps)
                self._dump_mps = self.dump_mps
            else:
                mps_abstract = ""
                self._dump_mps = None
            
            logger.info(f"step {len(self.evolve_times)-1} complete, time cost {time_cost}. {mps_abstract}")
            
            # dump
            if self._defined_output_path:
                try:
                    self.dump_dict()
                except IOError:  # never quit calculation because of IOError
                    logger.exception("dumping dict failed with IOError")
                dump_wall_time = datetime.now()
                logger.info(f"Dumping time cost {dump_wall_time - evolution_wall_time}")

        logger.info(f"{len(wall_times)-1} steps of evolution complete!")
        logger.info(
            "Normal termination. Time cost: %s" % (wall_times[-1] - wall_times[0])
        )
        return self

    def evolve_single_step(self, evolve_dt):
        """
        Evolve the mps for a single step with step size ``evolve_dt``.

        :return: new mps after the evolution
        """
        raise NotImplementedError

    def get_dump_dict(self):
        """
        Obtain calculated properties to dump in ``dict`` type.

        :return: return a (ordered) dict to dump as npz
        """
        raise NotImplementedError

    def dump_dict(self):
        if not self._defined_output_path:
            raise ValueError("Dump dir or job name not set")
        d = self.get_dump_dict()
        os.makedirs(self.dump_dir, exist_ok=True)
        file_path = os.path.join(self.dump_dir, self.job_name + ".npz")
        bak_path = file_path + ".bak"
        if os.path.exists(file_path):
            # in case of shutdown while dumping
            if os.path.exists(bak_path):
                os.remove(bak_path)
            os.rename(file_path, bak_path)

        np.savez(file_path, **d)

        if os.path.exists(bak_path):
            os.remove(bak_path)

        # dump_mps
        if self._dump_mps is not None:
            if self._dump_mps == "all":
                mps_path = os.path.join(self.dump_dir,
                        self.job_name+"_mps_"+str(len(self.evolve_times)-1) + ".npz")
            else:
                mps_path = os.path.join(self.dump_dir,
                        self.job_name+"_mps" + ".npz")
            self.latest_mps.dump(mps_path)
            

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
