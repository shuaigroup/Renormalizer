# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

import logging
import os
from enum import Enum
from collections import OrderedDict
from functools import partial

from scipy.linalg import logm

from renormalizer.mps import Mpo, Mps, MpDm, ThermalProp, load_thermal_state
from renormalizer.model import HolsteinModel
from renormalizer.utils import TdMpsJob, Quantity, CompressConfig, EvolveConfig

import numpy as np

logger = logging.getLogger(__name__)

EDGE_THRESHOLD = 1e-4


class InitElectron(Enum):
    """
    Available methods to prepare initial state of charge diffusion
    """
    fc = "franck-condon excitation"
    relaxed = "analytically relaxed phonon(s)"


class ChargeDiffusionDynamics(TdMpsJob):
    r"""
    Simulate charge diffusion dynamics by TD-DMRG. It is possible to obtain mobility from the simulation,
    but care must be taken to ensure that mean square displacement grows linearly with time.

    Args:
        model (:class:`~renormalizer.model.model.HolsteinModel`): system information.
            Currently only support :class:`~renormalizer.model.model.HolsteinModel`.
        temperature (:class:`~renormalizer.utils.quantity.Quantity`): simulation temperature.
            Default is zero temperature. For finite temperature charge dynamics, it is recommended to use
            thermal field dynamics and transform the Hamiltonian.
            See the documentation of :class:`~renormalizer.transport.spectral_function.SpectralFunctionZT`
            for more details.
        compress_config (:class:`~renormalizer.utils.configs.CompressConfig`): config when compressing MPS.
        evolve_config (:class:`~renormalizer.utils.configs.EvolveConfig`): config when evolving MPS.
        stop_at_edge (bool): whether stop when charge has diffused to the boundary of the system. Default is ``True``.
        init_electron (:class:`~renormalizer.transport.transport.InitElectron`):
            the method to prepare the initial state.
        rdm (bool): whether calculate reduced density matrix and k-space representation for the electron.
            Default is ``False`` because usually the calculation is time consuming.
            Using scheme 4 might partly solve the problem.
        dump_dir (str): the directory for logging and numerical result output.
            Also the directory from which to load previous thermal propagated initial state (if exists).
        job_name (str): the name of the calculation job which determines the file name of the logging and numerical result output.
            For thermal propagated initial state input/output the file name is appended with ``"_impdm.npz"``.

    Attributes:
        energies (np.ndarray): calculated energy of the states during the time evolution.
            Without dissipation or TD-Hartree the value should remain unchanged and to some extent
            can be used to measure the error during time evolution.
        r_square_array (np.ndarray): calculated mean square displacement
            :math:`\langle \psi | \hat r^2 | \psi \rangle - \langle \psi | \hat r | \psi \rangle^2`
            at each evolution time step.
        e_occupations_array (np.ndarray): calculated electron occupations in real space on each site for each evolution time step.
        ph_occupations_array (np.ndarray): calculated phonon occupations on each site for each evolution time step.
        reduced_density_matrices (list): calculated reduced density matrices of the electron for each evolution time step.
            Only available when ``rdm`` is set to ``True``.
        k_occupations_array (np.ndarray): calculated electron occupations in momentum (k) space
            on each site for each evolution time step. Only available when ``rdm`` is set to ``True``.
            The basis transformation is based on:

            .. math::
                | k \rangle = \sum_j e^{-ijk} | j \rangle

            where :math:`k` starts from :math:`-\pi` to :math:`\pi` with interval :math:`2\pi/N`.
            :math:`N` represents total number of electronic sites.
        coherent_length_array (np.ndarray): coherent length :math:`L` calculated for each evolution time step.

            .. math::
                L = \sum_{ij, i \neq j} | \rho_{ij} |

            where :math:`\rho` is the density matrix of the electron. Naturally this is only available when
            ``rdm`` is set to ``True``.

    """

    def __init__(
        self,
        model: HolsteinModel,
        temperature: Quantity = Quantity(0, "K"),
        compress_config: CompressConfig = None,
        evolve_config: EvolveConfig = None,
        stop_at_edge: bool = True,
        init_electron=InitElectron.relaxed,
        rdm: bool = False,
        dump_dir: str = None,
        job_name: str = None,
    ):
        self.model: HolsteinModel = model
        self.temperature = temperature
        self.mpo = None
        self.init_electron = init_electron
        if compress_config is None:
            self.compress_config: CompressConfig = CompressConfig()
        else:
            self.compress_config: CompressConfig = compress_config
        self.energies = []
        self.r_square_array = []
        self.e_occupations_array = []
        self.ph_occupations_array = []
        self.reduced_density_matrices = [] if rdm else None
        self.k_occupations_array = []
        # von Neumann entropy between e and ph
        self.eph_vn_entropy_array = []
        # entropy at each bond
        self.bond_vn_entropy_array = []
        self.coherent_length_array = []

        if dump_dir is not None and job_name is not None:
            self.thermal_dump_path = os.path.join(dump_dir, job_name + '_impdm.npz')
        else:
            self.thermal_dump_path = None

        super(ChargeDiffusionDynamics, self).__init__(evolve_config=evolve_config,
                                                      dump_dir=dump_dir, job_name=job_name)
        assert self.mpo is not None

        self.elocalex_arrays = []
        self.j_arrays = []
        self.custom_dump_info = OrderedDict()
        self.stop_at_edge = stop_at_edge

    @property
    def mol_num(self):
        return self.model.mol_num

    def create_electron_fc(self, gs_mp):
        center_mol_idx = self.mol_num // 2
        creation_operator = Mpo.onsite(
            self.model, r"a^\dagger", dof_set={center_mol_idx}
        )
        mps = creation_operator.apply(gs_mp)
        return mps

    def create_electron_relaxed(self, gs_mp):
        assert np.allclose(gs_mp.bond_dims, np.ones_like(gs_mp.bond_dims))
        center_mol_idx = self.mol_num // 2
        center_mol = self.model[center_mol_idx]
        # start from phonon
        for i, ph in enumerate(center_mol.ph_list):
            idx = self.model.order[(center_mol_idx, i)]
            mt = gs_mp[idx][0, ..., 0].array
            evecs = ph.get_displacement_evecs()
            mt = evecs.dot(mt)
            logger.debug(f"relaxed mt: {mt}")
            gs_mp[idx] = mt.reshape([1] + list(mt.shape) + [1])

        creation_operator = Mpo.onsite(
            self.model, r"a^\dagger", dof_set={center_mol_idx}
        )
        mps = creation_operator.apply(gs_mp)
        return mps

    def create_electron(self, gs_mp):
        method_mapping = {
            InitElectron.fc: self.create_electron_fc,
            InitElectron.relaxed: self.create_electron_relaxed,
        }
        logger.info(f"Creating electron using {self.init_electron}")
        return method_mapping[self.init_electron](gs_mp)

    def init_mps(self):
        tentative_mpo = Mpo(self.model)
        if self.temperature == 0:
            gs_mp = Mps.ground_state(self.model, max_entangled=False)
        else:
            if self.thermal_dump_path is not None:
                gs_mp = load_thermal_state(self.model, self.thermal_dump_path)
            else:
                gs_mp = None
            if gs_mp is None:
                gs_mp = MpDm.max_entangled_gs(self.model)
                tp = ThermalProp(gs_mp, exact=True, space="GS")
                tp.evolve(None, max(20, len(gs_mp)), self.temperature.to_beta() / 2j)
                gs_mp = tp.latest_mps
                if self.thermal_dump_path is not None:
                    gs_mp.dump(self.thermal_dump_path)
        init_mp = self.create_electron(gs_mp)
        energy = Quantity(init_mp.expectation(tentative_mpo))
        self.mpo = Mpo(self.model, offset=energy)
        logger.info(f"mpo bond dims: {self.mpo.bond_dims}")
        logger.info(f"mpo physical dims: {self.mpo.pbond_list}")
        init_mp.evolve_config = self.evolve_config
        init_mp.compress_config = self.compress_config
        if self.evolve_config.is_tdvp:
            init_mp = init_mp.expand_bond_dimension(self.mpo)
        init_mp.canonicalise()
        return init_mp

    def process_mps(self, mps):
        new_energy = mps.expectation(self.mpo)
        self.energies.append(new_energy)
        logger.debug(f"Energy: {new_energy}")

        if self.reduced_density_matrices is not None:
            logger.debug("Calculating reduced density matrix")
            rdm = mps.calc_edof_rdm()
            logger.debug("Calculate reduced density matrix finished")
            self.reduced_density_matrices.append(rdm)

            # k_space transform matrix
            n = len(self.model)
            assert rdm.shape == (n, n)
            transform = np.exp(-1j * (np.arange(-n, n, 2)/n * np.pi).reshape(-1, 1) * np.arange(0, n).reshape(1, -1)) / np.sqrt(n)
            k = np.diag(transform @ rdm @ transform.conj().T).real
            self.k_occupations_array.append(k)

            # von Neumann entropy
            entropy = -np.trace(rdm @ logm(rdm))
            self.eph_vn_entropy_array.append(entropy)

            self.coherent_length_array.append(np.abs(rdm).sum() - np.trace(rdm).real)

        else:
            rdm = None

        if rdm is not None:
            e_occupations = np.diag(rdm).real
        else:
            e_occupations = mps.e_occupations
        self.e_occupations_array.append(e_occupations)
        self.r_square_array.append(calc_r_square(e_occupations))
        self.ph_occupations_array.append(mps.ph_occupations)
        logger.info(f"e occupations: {self.e_occupations_array[-1]}")

        bond_vn_entropy = mps.calc_bond_entropy()
        logger.info(f"bond entropy: {bond_vn_entropy}")
        self.bond_vn_entropy_array.append(bond_vn_entropy)

    def evolve_single_step(self, evolve_dt):
        old_mps = self.latest_mps
        new_mps = old_mps.evolve(self.mpo, evolve_dt)
        return new_mps

    def stop_evolve_criteria(self):
        # electron has moved to the edge
        return self.stop_at_edge and EDGE_THRESHOLD < self.e_occupations_array[-1][0]

    def get_dump_dict(self):
        dump_dict = OrderedDict()
        dump_dict["mol list"] = self.model.to_dict()
        dump_dict["tempearture"] = self.temperature.as_au()
        dump_dict["total time"] = self.evolve_times[-1]
        dump_dict["other info"] = self.custom_dump_info
        # make np array json serializable
        dump_dict["r square array"] = self.r_square_array
        dump_dict["electron occupations array"] = self.e_occupations_array
        dump_dict["phonon occupations array"] = self.ph_occupations_array
        dump_dict["k occupations array"] = self.k_occupations_array
        dump_dict["eph entropy"] = self.eph_vn_entropy_array
        dump_dict["bond entropy"] = self.bond_vn_entropy_array
        dump_dict["coherent length array"] = self.coherent_length_array
        if self.reduced_density_matrices:
            dump_dict["reduced density matrices"] = self.reduced_density_matrices
        dump_dict["time series"] = list(self.evolve_times)
        return dump_dict

    def is_similar(self, other: "ChargeDiffusionDynamics", rtol=1e-3):
        all_close_with_tol = partial(np.allclose, rtol=rtol, atol=1e-3)
        if len(self.evolve_times) != len(other.evolve_times):
            return False
        attrs = [
            "evolve_times",
            "r_square_array",
            "energies",
            "e_occupations_array",
            "ph_occupations_array",
            "coherent_length_array",
        ]
        for attr in attrs:
            s = getattr(self, attr)
            o = getattr(other, attr)
            if not all_close_with_tol(s, o):
                return False
        return True


def calc_r_square(e_occupations):
    r_list = np.arange(0, len(e_occupations))
    if np.allclose(e_occupations, np.zeros_like(e_occupations)):
        return 0
    r_mean_square = np.average(r_list, weights=e_occupations) ** 2
    mean_r_square = np.average(r_list ** 2, weights=e_occupations)
    return float(mean_r_square - r_mean_square)
