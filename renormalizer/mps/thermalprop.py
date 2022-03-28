import logging

import numpy as np

from renormalizer.model import Model
from renormalizer.mps import MpDm, Mpo
from renormalizer.utils import TdMpsJob, Quantity, EvolveConfig
from renormalizer.property import Property

logger = logging.getLogger(__name__)


class ThermalProp(TdMpsJob):
    r"""
    Thermally Propagate initial matrix product density operator (:class:`~renormalizer.mps.MpDm`) in imaginary time.

    Args:
        init_mpdm (:class:`~renormalizer.mps.MpDm`): the initial density matrix to be propagated. Usually identity.
        h_mpo_model (:class:`~renormalizer.model.model.Model`): Model for the system Hamiltonian.
            Default is the same with ``init_mpdm.model``.
        exact (bool): whether propagate assuming Hamiltonian is local
            :math:`\hat H = \sum_i \hat H_i = \sum_{in} \omega_{in} b^\dagger_{in} b_{in}` and
            exact propagation is possible through :math:`e^{xH} = e^{xh_1}e^{xh_2} \cdots e^{xh_n}`.
            If set to ``True``, properties such as occupations are not calculated during
            time evolution for better efficiency.
        space (str): the space of exact propagation. Possible options are ``"GS"`` or ``"EX"``.
            If set to ``"GS"``, then the exact propagation is performed in zero exciton space.
            If set to ``"EX"``, then the exact propagation is performed in one exciton space,
            i.e. the vibrations are regarded as displaced oscillators.
        evolve_config (:class:`~renormalizer.utils.EvolveConfig`): config when evolving the MpDm in imaginary time.
        dump_mps (str): if dump mps when dumping, "all", "one", None; Default to None
        dump_dir (str): the directory for logging and numerical result output.
        job_name (str): the name of the calculation job which determines the file name of the logging and numerical result output.
        properties (:class:`~renormalizer.property.Property`) calculate other properties with interface in Property
    """
    def __init__(
        self,
        init_mpdm: MpDm,
        h_mpo_model: Model = None,
        exact: bool = False,
        space: str = "GS",
        evolve_config: EvolveConfig = None,
        dump_mps: bool = None, 
        dump_dir: str = None,
        job_name: str = None,
        properties: Property = None,
        auto_expand: bool = True,
    ):
        self.init_mpdm: MpDm = init_mpdm.canonicalise()
        if h_mpo_model is None:
            h_mpo_model = self.init_mpdm.model
        self.h_mpo = Mpo(h_mpo_model)
        logger.info(f"Bond dim of h_mpo: {self.h_mpo.bond_dims}")
        self.exact = exact
        assert space in ["GS", "EX"]
        self.space = space
        self.energies = []
        self._e_occupations_array = []
        self._ph_occupations_array = []
        self._vn_entropy_array = []
        self.properties = properties
        self.auto_expand = auto_expand

        super().__init__(evolve_config=evolve_config, dump_mps=dump_mps, dump_dir=dump_dir,
                job_name=job_name)

    def init_mps(self):
        self.init_mpdm.evolve_config = self.evolve_config
        if self.evolve_config.is_tdvp and self.auto_expand:
            self.init_mpdm = self.init_mpdm.expand_bond_dimension(self.h_mpo)
        return self.init_mpdm

    def process_mps(self, mps):
        new_energy = mps.expectation(self.h_mpo)
        self.energies.append(new_energy)
        if self.exact:
            # skip the fuss for efficiency
            return
        for attr_str in ["e_occupations", "ph_occupations"]:
            attr = getattr(mps, attr_str)
            logger.info(f"{attr_str}: {attr}")
            self_array = getattr(self, f"_{attr_str}_array")
            self_array.append(attr)
        vn_entropy = mps.calc_bond_entropy()
        self._vn_entropy_array.append(vn_entropy)
        logger.info(f"vn entropy: {vn_entropy}")
        logger.info(
            f"Energy: {new_energy}, total electron: {self._e_occupations_array[-1].sum()}"
        )
        
        # calculate other properties defined in Property
        if self.properties is not None:
            self.properties.calc_properties(mps)

    def evolve_exact(self, old_mpdm: MpDm, evolve_dt):
        MPOprop = Mpo.exact_propagator(
            old_mpdm.model, evolve_dt.imag, space=self.space, shift=-self.energies[-1]
        )
        new_mpdm = MPOprop.apply(old_mpdm, canonicalise=True)
        # partition function can't be obtained. It's not practical anyway.
        # The function is too large to be fit into float64 even float128
        new_mpdm.normalize("mps_and_coeff")
        return new_mpdm

    def evolve_prop(self, old_mpdm, evolve_dt):
        h_mpo = Mpo(self.h_mpo.model, offset=Quantity(self.energies[-1]))
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

    @property
    def vn_entropy_array(self):
        return np.array(self._vn_entropy_array)

    def get_dump_dict(self):
        dump_dict = dict()
        dump_dict["time series"] = [-t.imag for t in self.evolve_times]
        dump_dict["energies"] = self.energies
        dump_dict["electron occupations array"] = self.e_occupations_array.tolist()
        dump_dict["phonon occupations array"] = self.ph_occupations_array.tolist()
        dump_dict["vn entropy array"] = self.vn_entropy_array.tolist()
        
        if self.properties is not None:
            for prop_str in self.properties.prop_res.keys():
                dump_dict[prop_str] = self.properties.prop_res[prop_str]

        return dump_dict


def load_thermal_state(model, path: str):
    """
    Load thermal propagated state from disk. Return None if the file is not found.

    Args:
        model (:class:`MolList`): system information
        path (str): the path to load thermal state from. Should be an numpy ``.npz`` file.
    Returns: Loaded MpDm
    """
    try:
        logger.info(f"Try load from {path}")
        mpdm = MpDm.load(model, path)
        logger.info(f"Init mpdm loaded: {mpdm}")
    except FileNotFoundError:
        logger.info(f"No file found in {path}")
        mpdm = None

    return mpdm
