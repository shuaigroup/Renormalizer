# -*- coding: utf-8 -*-

import logging

from renormalizer.mps.backend import np
from renormalizer.mps import Mpo, Mps
from renormalizer.utils import TdMpsJob, Quantity, EvolveConfig, CompressConfig
from renormalizer.model import Model, TI1DModel


logger = logging.getLogger(__name__)



class SpectralFunctionZT(TdMpsJob):
    r"""
    Calculate one-particle retarded Green's function at zero temperature for translational invariant one-dimensional model:

    .. math::
        iG_{ij}(t) = \langle 0 |c_i(t) c^\dagger_j | 0 \rangle

    The value of the array is stored with key ``"G array"`` in the dumped output file.
    Because of the translational invariance, there are only two dimension in this array.
    The first dimension is :math:`t` and the second dimension is :math:`|i-j|`.
    In k-space, :math:`iG_{ij}` is transformed to:

    .. math::
        iG_k(t) = \langle 0 |c_k(t) c^\dagger_k | 0 \rangle

    and :math:`G_k(t)` is saved with key ``"Gk array"`` in the dumped output file.
    Fourier transformation of :math:`G_k(t)` results in the spectral function:

    .. math::
        A(k, \omega) = -\frac{1}{\pi} \rm{Im} \int_0^\infty dt e^{i \omega t} G_k(t)

    However, this value is not calculated directly in this class, since usually an broadening to :math:`G_k(t)`
    is required and an optimal range of :math:`\omega` can not be determined without additional knowledge.

    For finite temperature spectral function, it is recommended to use
    thermal field dynamics and transform the Hamiltonian.
    An example is included in the test case.
    See J. Chem.Phys.145, 224101 (2016) and
    Phys. Rev. Lett.123, 090402 (2019) for details.

    Parameters
    ==========
    model : :class:`~renormalizer.model.TI1DModel`
        system information. In principle should use :class:`~renormalizer.model.TI1DModel`.
        Using custom :class:`~renormalizer.model.Model` is possible however
        in this case the user should be responsible to ensure the translational invariance.
        It is recommended to set the number of the electronic DoFs to be an even number
        so that the :math:`k=\pi` point is well defined.
    compress_config : :class:`~renormalizer.utils.configs.CompressConfig`
        config when compressing MPS.
    evolve_config : :class:`~renormalizer.utils.configs.EvolveConfig`
        config when carrying out real time propagation.
    dump_dir : str
        the directory for logging and numerical result output.
    job_name : str
        the name of the calculation job which determines the file name of the logging and numerical result output.
    """

    def __init__(
            self,
            model: TI1DModel,
            compress_config: CompressConfig = None,
            evolve_config: EvolveConfig = None,
            dump_dir: str = None,
            job_name: str = None,
    ):
        self.model: TI1DModel = model
        self.compress_config = compress_config
        if self.compress_config is None:
            self.compress_config = CompressConfig()
        # electron-addition Green's function at different $t$ assuming translational invariance
        self._G_array = []
        self.e_occupations_array = []
        self.temperature = Quantity(0)
        super().__init__(evolve_config, dump_dir, job_name)

    @property
    def G_array(self):
        """
        :math:`G_{ij}(t)` as a two dimensional array.
        The first dimension is :math:`t` and the second dimension is :math:`|i-j|`.

        Returns
        =======
        G_array : np.ndarray
            The Green's function array
        """
        return np.array(self._G_array)

    def init_mps(self):
        creation_oper = Mpo.onsite(self.model, r"a^\dagger", dof_set={self.model.e_dofs[0]})
        gs = Mps.ground_state(self.model, False)
        self.h_mpo = Mpo(self.model, offset=Quantity(gs.expectation(Mpo(self.model))))
        a_ket = creation_oper.apply(gs, canonicalise=True)
        a_ket.compress_config = self.compress_config
        a_ket.evolve_config = self.evolve_config
        a_ket.normalize("mps_norm_to_coeff")
        if self.evolve_config.is_tdvp:
            a_ket = a_ket.expand_bond_dimension(self.h_mpo)
        return (gs, a_ket)

    def process_mps(self, mps):
        key = "a"
        if key not in self.model.mpos:
            a_opers = [Mpo.onsite(self.model, "a", dof_set={dof}) for dof in self.model.e_dofs]
            self.model.mpos[key] = a_opers
        else:
            a_opers = self.model.mpos[key]

        a_bra_mpo, a_ket_mpo = mps
        G = a_ket_mpo.expectations(a_opers, a_bra_mpo.conj()) / 1j
        self._G_array.append(G)
        self.e_occupations_array.append(a_ket_mpo.e_occupations)

    def evolve_single_step(self, evolve_dt):
        prev_bra_mpdm, prev_ket_mpdm = self.latest_mps
        latest_ket_mpdm = prev_ket_mpdm.evolve(self.h_mpo, evolve_dt)
        return (prev_bra_mpdm, latest_ket_mpdm)

    def get_dump_dict(self):
        dump_dict = dict()
        dump_dict['temperature'] = self.temperature.as_au()
        dump_dict['time series'] = self.evolve_times
        dump_dict["G array"] = self.G_array
        ne = self.model.n_edofs
        kpoints_distance = (2 * np.pi) / ne
        n_kpoints = ne // 2 + 1
        ka = (np.arange(n_kpoints) * kpoints_distance).reshape(1, 1, -1)
        ijdiff = np.arange(ne).reshape(1, -1, 1)
        # Green's function in k space
        dump_dict["Gk array"] = np.sum(self.G_array.reshape(-1, ne, 1) * np.exp(1j * ka * ijdiff), axis=1)
        dump_dict["electron occupations array"] = self.e_occupations_array
        return dump_dict
