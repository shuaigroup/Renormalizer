from renormalizer.utils import TdMpsJob, CompressConfig, EvolveConfig
from renormalizer.mps import Mps, Mpo
from renormalizer.property import Property
from renormalizer.model import MolList2

import logging
import numpy as np

logger = logging.getLogger(__name__)


class VibronicModelDynamics(TdMpsJob):
    r"""
    Vibronic Hamiltonian Dynamics
    Args:
        mol_list (:class:`~renormalizer.model.MolList2`)

    """
    
    def __init__(
            self, 
            mol_list: MolList2, 
            compress_config: CompressConfig = None,
            evolve_config: EvolveConfig = None,
            h_mpo = None,
            mps0 = None,
            properties: Property = None,
            init_condition = None, 
            dump_mps: bool = False,
            dump_dir: str = None,
            job_name: str = None,
        ):

        self.mol_list = mol_list
        
        if compress_config is None:
            self.compress_config = CompressConfig()
        else:
            self.compress_config = compress_config
        
        if h_mpo is None:
            self.h_mpo = Mpo(mol_list)
        else:
            self.h_mpo = h_mpo

        self.mps0 = mps0
        self.init_condition = init_condition
        self.properties = properties
        
        self.e_occupations_array = []
        self.autocorr_array = []
        self.energies = []
        self.autocorr_time = []

        super().__init__(evolve_config=evolve_config, dump_mps=dump_mps, dump_dir=dump_dir,
                job_name=job_name)
    

    def init_mps(self):
        if self.mps0 is None:
            assert self.init_condition is not None
            init_mp = Mps.hartree_product_state(self.mol_list, self.init_condition)
            self.mps0 = init_mp.copy()
        else:
            init_mp = self.mps0.copy()
        init_mp.compress_config = self.compress_config
        init_mp.evolve_config = self.evolve_config
        init_mp.mol_list = self.mol_list
        if self.evolve_config.is_tdvp:
            init_mp = init_mp.expand_bond_dimension(self.h_mpo)
        return init_mp

    def evolve_single_step(self, evolve_dt):
        old_mps = self.latest_mps
        mpo = self.h_mpo
        new_mps = old_mps.evolve(mpo, evolve_dt)
        
        return new_mps

    def process_mps(self, mps):
        # energy
        new_energy = mps.expectation(self.h_mpo)
        self.energies.append(new_energy)
        logger.debug(f"Energy: {new_energy}")
        # electron population
        e_occupations = mps.e_occupations
        self.e_occupations_array.append(e_occupations)
        logger.debug(f"e occupations: {self.e_occupations_array[-1]}")
        # autocorrelation function
        if self.mps0.is_complex:
            autocorr = mps.conj().dot(self.mps0)
            self.autocorr_array.append(autocorr)
            self.autocorr_time.append(self.evolve_times[-1])
        else:
            # make sure the latest mps is the last step mps and not ruined by
            # evolve_single_step
            # in tdmps latest_mps = new_mps after process_mps
            if not np.allclose(self.evolve_times[-1], 0):
                autocorr = mps.dot(self.latest_mps)
                self.autocorr_array.append(autocorr)
                self.autocorr_time.append(self.evolve_times[-1] + self.evolve_times[-2])
            autocorr = mps.dot(mps)
            self.autocorr_array.append(autocorr)
            self.autocorr_time.append(self.evolve_times[-1] + self.evolve_times[-1])

    def get_dump_dict(self):
        """
        :return: return a (ordered) dict to dump as json or npz
        """
        dump_dict = dict()
        dump_dict["time series"] = list(self.evolve_times)
        dump_dict["electron occupations array"] = self.e_occupations_array
        dump_dict["autocorrelation function"] = self.autocorr_array
        dump_dict["autocorrelation time"] = self.autocorr_time
        dump_dict["energy"] = self.energies

        return dump_dict
