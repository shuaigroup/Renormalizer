import logging

import numpy as np

from renormalizer.mps import MpDm, Mpo
from renormalizer.mps.tdh import unitary_propagation
from renormalizer.utils import TdMpsJob, Quantity
from renormalizer.utils.utils import cast_float


logger = logging.getLogger(__name__)


class ThermalProp(TdMpsJob):

    def __init__(self, init_mpdm, h_mpo, exact=False, space="GS", evolve_config=None, dump_dir=None,
                 job_name=None):
        self.init_mpdm: MpDm = init_mpdm
        self.h_mpo = h_mpo
        self.exact = exact
        assert space in ["GS", "EX"]
        self.space = space
        self.energies = []
        self._e_occupations_array = []
        self._ph_occupations_array = []
        super().__init__(evolve_config, dump_dir, job_name)

    def init_mps(self):
        self.init_mpdm.evolve_config = self.evolve_config
        self.energies.append(self.init_mpdm.expectation(self.h_mpo))
        if self.evolve_config.is_tdvp:
            self.init_mpdm = self.init_mpdm.expand_bond_dimension(self.h_mpo)
        return self.init_mpdm

    def process_mps(self, mps):
        new_energy = mps.expectation(self.h_mpo)
        self.energies.append(new_energy)
        logger.info(f"Energy: {new_energy}, total electron: {mps.e_occupations.sum()}")
        for attr_str in ["e_occupations", "ph_occupations"]:
            attr = getattr(mps, attr_str)
            logger.info(f"{attr_str}: {attr}")
            self_array = getattr(self, f"_{attr_str}_array")
            self_array.append(attr)

    def evolve_exact(self, old_mpdm, evolve_dt):
        MPOprop, HAM, Etot = old_mpdm.hybrid_exact_propagator(
            self.h_mpo, evolve_dt.imag, space=self.space
        )
        new_mpdm = MPOprop.apply(old_mpdm)
        unitary_propagation(new_mpdm.tdh_wfns, HAM, Etot, evolve_dt)
        # partition function can't be obtained. It's not practical anyway.
        # The function is too large to be fit into float64 even float128
        new_mpdm.normalize(1.0)
        # the mpdm may not be canonicalised due to distributed scaling. It's not wise to do
        # so currently because scheme4 might have empty matrices
        # new_mpdm.canonicalise()
        return new_mpdm

    def evolve_prop(self, old_mpdm, evolve_dt):
        h_mpo = Mpo(self.h_mpo.mol_list, offset=Quantity(self.energies[-1]))
        return old_mpdm.evolve(h_mpo, evolve_dt)

    def evolve_single_step(self, evolve_dt):
        old_mpdm = self.latest_mps
        if self.exact:
            new_mpdm = self.evolve_exact(old_mpdm, evolve_dt)
        else:
            new_mpdm = self.evolve_prop(old_mpdm, evolve_dt)
        return new_mpdm

    def evolve(self, evolve_dt=None, nsteps=None, evolve_time=None):
        if evolve_dt is not None:
            assert np.iscomplex(evolve_dt) and evolve_dt.imag < 0
        if evolve_time is not None:
            assert np.iscomplex(evolve_time) and evolve_time.imag < 0
        super().evolve(evolve_dt, nsteps, evolve_time)

    @property
    def e_occupations_array(self):
        return np.array(self._e_occupations_array)

    @property
    def ph_occupations_array(self):
        return np.array(self._ph_occupations_array)

    def get_dump_dict(self):
        dump_dict = dict()
        dump_dict["time series"] = [-t.imag for t in self.evolve_times]
        dump_dict["energies"] = self.energies
        dump_dict["electron occupations array"] = cast_float(self.e_occupations_array)
        dump_dict["phonon occupations array"] = cast_float(self.ph_occupations_array)
        return dump_dict
