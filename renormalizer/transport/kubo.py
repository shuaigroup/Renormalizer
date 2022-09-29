# -*- coding: utf-8 -*-

import logging
import os

import scipy.integrate

from renormalizer.mps import MpDm, Mpo, BraKetPair, ThermalProp, load_thermal_state
from renormalizer.mps.backend import np
from renormalizer.utils.constant import mobility2au
from renormalizer.utils import TdMpsJob, Quantity, EvolveConfig, CompressConfig
from renormalizer.model import Model
from renormalizer.property import Property

logger = logging.getLogger(__name__)


class TransportKubo(TdMpsJob):
    r"""
    Calculate mobility via Green-Kubo formula:

        .. math::
            \mu = \frac{1}{k_B T} \int_0^\infty dt \langle \hat j (t) \hat j(0) \rangle
            = \frac{1}{k_B T} \int_0^\infty dt C(t)

    where

        .. math::
           \hat j = -\frac{i}{\hbar}[\hat P, \hat H]

    and :math:`\hat P = e_0 \sum_m R_m a^\dagger_m a_m` is the polarization operator.

    .. note::
        In principle :math:`\hat H` can take any form, however only Holstein-Peierls Hamiltonian is well tested.

    More explicitly, :math:`C(t)` has the form:

        .. math::
            C(t) = \textrm{Tr}\{\rho(T) e^{i \hat H t} \hat j(0) e^{- i \hat H t} \hat j (0)\}

    where we have assumed :math:`\rho(T)` is normalized
    (i.e. it is divided by the partition function :math:`\textrm{Tr}\{\rho(T)\}`).

    In terms of practical implementation, it is ideal if :math:`\rho(T)` is split into two parts
    to (hopefully) speed up calculation and minimize time evolution error:

        .. math::
            \begin{align}
             C(t) & = \textrm{Tr}\{\rho(T) e^{i \hat H t} \hat j(0) e^{- i \hat H t} \hat j(0)\} \\
                  & = \textrm{Tr}\{e^{-\beta \hat H} e^{i \hat H t} \hat j(0) e^{- i \hat H t} \hat j(0)\} \\
                  & = \textrm{Tr}\{e^{-\beta \hat H / 2} e^{i \hat H t} \hat j(0) e^{- i \hat H t} \hat j(0) e^{-\beta \hat H / 2}\}
            \end{align}

    In this class, imaginary time propagation from infinite temperature to :math:`\beta/2` is firstly carried out
    to obtain :math:`e^{-\beta \hat H / 2}`, and then real time propagation is carried out for :math:`e^{-\beta \hat H / 2}`
    and :math:`\hat J(0) e^{-\beta \hat H / 2}` respectively to obtain :math:`e^{-\beta \hat H / 2} e^{i \hat H t}`
    and :math:`e^{- i \hat H t} \hat j(0) e^{-\beta \hat H / 2}`. The correlation function at time :math:`t` can thus
    be calculated via expectation calculation.

    .. note::
        Although the class is able to carry out imaginary time propagation, in practice for large scale calculation
        it is usually preferable to carry out imaginary time propagation in another job and load the dumped initial
        state directly in this class.

    Args:
        model (:class:`~renormalizer.model.Model`): system information.
        temperature (:class:`~renormalizer.utils.quantity.Quantity`): simulation temperature.
            Zero temperature is not supported.
        distance_matrix (:class:`np.ndarray`): two-dimensional array :math:`D_{ij} = P_i - P_j` representing
            distance between the :math:`i` th electronic degree of freedom and the :math:`j` th degree of freedom.
            The index is the same with ``model.e_dofs``.
            The parameter takes the role of :math:`\hat P` and can better handle periodic boundary condition.
            The default value is ``None`` in which case the distance matrix is constructed assuming the system
            is a one-dimensional chain.

            .. note::
                The construction of the matrix should be taken with great care if periodic boundary condition
                is applied. Take a one-dimensional chain as an example, the distance between the leftmost site
                and the rightmost site is :math:`\pm R` where :math:`R` is the intersite distance,
                rather than :math:`\pm (N-1)R` where :math:`N` is the total number of electronic degrees of freedom.
        insteps (int): steps for imaginary time propagation.
        ievolve_config (:class:`~renormalizer.utils.configs.EvolveConfig`): config when carrying out imaginary time propagation.
        compress_config (:class:`~renormalizer.utils.configs.CompressConfig`): config when compressing MPS.
            Note that if TDVP based methods are chosen for time evolution, compression is necessary
            when :math:`\hat j` is applied on :math:`e^{-\beta \hat H / 2}` but no compression is carried out
            for :math:`e^{-\beta \hat H / 2} e^{i \hat H t}`.
        evolve_config (:class:`~renormalizer.utils.configs.EvolveConfig`): config when carrying out real time propagation.
        dump_dir (str): the directory for logging and numerical result output.
        job_name (str): the name of the calculation job which determines the file name of the logging and numerical result output.
        thermal_dump_path (str): the path to load previously thermal propagated initial state if it exists or the path
            to dump the thermal state if it does not exist. The default value is determined by appending ``"_impdm.npz"``
            to the path joined by ``dump_dir`` and ``job_name``.
        properties (:class:`~renormalizer.property.property.Property`): other properties to calculate during real time evolution.
            Currently only supports Holstein model.
    """
    def __init__(self, model: Model, temperature: Quantity, distance_matrix: np.ndarray = None,
                 insteps: int=1, ievolve_config=None, compress_config=None,
                 evolve_config=None, dump_dir: str=None, job_name: str=None,
                 thermal_dump_path: str=None, properties: Property = None):
        self.model = model
        self.distance_matrix = distance_matrix
        self.h_mpo = Mpo(model)
        logger.info(f"Bond dim of h_mpo: {self.h_mpo.bond_dims}")
        self._construct_current_operator()
        if temperature == 0:
            raise ValueError("Can't set temperature to 0.")
        self.temperature = temperature

        # imaginary time evolution config
        if ievolve_config is None:
            self.ievolve_config = EvolveConfig()
            if insteps is None:
                self.ievolve_config.adaptive = True
                # start from a small step
                self.ievolve_config.guess_dt = temperature.to_beta() / 1e5j
                insteps = 1
        else:
            self.ievolve_config = ievolve_config
        self.insteps = insteps

        if compress_config is None:
            logger.debug("using default compress config")
            self.compress_config = CompressConfig()
        else:
            self.compress_config = compress_config

        if thermal_dump_path is not None:
            self.thermal_dump_path = thermal_dump_path
        elif dump_dir is not None and job_name is not None:
            self.thermal_dump_path = os.path.join(dump_dir, job_name + '_impdm.npz')
        else:
            self.thermal_dump_path = None

        self.properties = properties
        self._auto_corr = []
        self._auto_corr_deomposition = []
        super().__init__(evolve_config=evolve_config, dump_dir=dump_dir,
                job_name=job_name)

    def _construct_current_operator(self):
        # Construct current operator. The operator is taken to be real as an optimization.
        logger.info("constructing current operator ")

        mol_num = self.model.n_edofs
        ham_terms = self.model.ham_terms

        if self.distance_matrix is None:
            logger.info("Constructing distance matrix based on a periodic one-dimension chain.")
            self.distance_matrix = np.arange(mol_num).reshape(-1, 1) - np.arange(mol_num).reshape(1, -1)
            self.distance_matrix[0][-1] = 1
            self.distance_matrix[-1][0] = -1

        # current caused by pure eletronic coupling
        holstein_current_terms = []
        # current related to phonons
        peierls_current_terms = []
        # loop through the Hamiltonian to construct current operator
        for ham_op in ham_terms:
            # find out terms that contains two electron operators
            # idx of the dof for the model
            dof_op_idx1 = dof_op_idx2 = None
            for dof_idx, dof_name in enumerate(ham_op.dofs):
                site_idx = self.model.dof_to_siteidx[dof_name]
                if self.model.basis[site_idx].is_electron:
                    e_idx = self.model.e_dofs.index(dof_name)
                    if dof_op_idx1 is None:
                        dof_op_idx1 = dof_idx
                        e_idx1 = e_idx
                    elif dof_op_idx2 is None:
                        dof_op_idx2 = dof_idx
                        e_idx2 = e_idx
                    else:
                        raise ValueError(f"The model contains three-electron (or more complex) operator {ham_op}")
                del dof_idx, dof_name
            # two electron operators not found. Not relevant to the current operator
            if dof_op_idx1 is None or dof_op_idx2 is None:
                continue
            # electron operator on the same site. Not relevant to the current operator.
            if e_idx1 == e_idx2:
                continue
            # two electron operators found. Relevant to the current operator
            # perform a bunch of sanity check
            # at most 3 dofs are involved. More complex cases are probably supported but not tested
            if len(ham_op.dofs) not in (2, 3):
                raise NotImplementedError("Complex vibration potential not implemented")

            # check linear coupling. More complex cases are probably supported but not tested.
            if len(ham_op.dofs) == 3:
                # total term idx should be 0 + 1 + 2 = 3
                phonon_dof_idx = 3 - dof_op_idx1 - dof_op_idx2
                assert ham_op.split_symbol[phonon_dof_idx] in (r"b^\dagger+b", "x")
            symbol1, symbol2 = ham_op.split_symbol[dof_op_idx1], ham_op.split_symbol[dof_op_idx2]
            if not {symbol1, symbol2} == {r"a^\dagger", "a"}:
                raise ValueError(f"Unknown symbol: {symbol1}, {symbol2}")

            # translate the term in the Hamiltonian into the term in the current operator
            if symbol1 == r"a^\dagger":
                factor = self.distance_matrix[e_idx1][e_idx2]
            else:
                factor = self.distance_matrix[e_idx2][e_idx1]

            current_op = ham_op * factor
            # Holstein terms
            if len(ham_op.dofs) == 2:
                holstein_current_terms.append(current_op)
            # Peierls terms
            else:
                peierls_current_terms.append(current_op)

        self.j_oper = Mpo(self.model, holstein_current_terms)
        logger.info(f"current operator bond dim: {self.j_oper.bond_dims}")
        if len(peierls_current_terms) != 0:
            self.j_oper2  = Mpo(self.model, peierls_current_terms)
            logger.info(f"Peierls coupling induced current operator bond dim: {self.j_oper2.bond_dims}")
        else:
            self.j_oper2 = None

    def init_mps(self):
        # first try to load
        if self.thermal_dump_path is not None:
            mpdm = load_thermal_state(self.model, self.thermal_dump_path)
        else:
            mpdm = None
        # then try to calculate
        if mpdm is None:
            i_mpdm = MpDm.max_entangled_ex(self.model)
            i_mpdm.compress_config = self.compress_config
            if self.job_name is None:
                job_name = None
            else:
                job_name = self.job_name + "_thermal_prop"
            tp = ThermalProp(i_mpdm, evolve_config=self.ievolve_config, dump_dir=self.dump_dir, job_name=job_name)
            # only propagate half beta
            tp.evolve(None, self.insteps, self.temperature.to_beta() / 2j)
            mpdm = tp.latest_mps
            if self.thermal_dump_path is not None:
                mpdm.dump(self.thermal_dump_path)
        mpdm.compress_config = self.compress_config
        e = mpdm.expectation(self.h_mpo)
        self.h_mpo = Mpo(self.model, offset=Quantity(e))
        mpdm.evolve_config = self.evolve_config
        logger.debug("Applying current operator")
        ket_mpdm = self.j_oper.contract(mpdm).normalize("mps_norm_to_coeff")
        bra_mpdm = mpdm.copy()
        if self.j_oper2 is None:
            return BraKetPair(bra_mpdm, ket_mpdm, self.j_oper)
        else:
            logger.debug("Applying the second current operator")
            ket_mpdm2 = self.j_oper2.contract(mpdm).normalize("mps_norm_to_coeff")
            return BraKetPair(bra_mpdm, ket_mpdm, self.j_oper), BraKetPair(bra_mpdm, ket_mpdm2, self.j_oper2)

    def process_mps(self, mps):
        # add the negative sign because `self.j_oper` is taken to be real
        if self.j_oper2 is None:
            self._auto_corr.append(-mps.ft)
            # calculate other properties defined in Property
            if self.properties is not None:
                self.properties.calc_properties_braketpair(mps)
        else:
            (bra_mpdm, ket_mpdm), (bra_mpdm, ket_mpdm2) = mps
            # <J_1(t) J_1(0)>
            ft1 = -BraKetPair(bra_mpdm, ket_mpdm, self.j_oper).ft
            # <J_1(t) J_2(0)>
            ft2 = -BraKetPair(bra_mpdm, ket_mpdm2, self.j_oper).ft
            # <J_2(t) J_1(0)>
            ft3 = -BraKetPair(bra_mpdm, ket_mpdm, self.j_oper2).ft
            # <J_2(t) J_2(0)>
            ft4 = -BraKetPair(bra_mpdm, ket_mpdm2, self.j_oper2).ft
            self._auto_corr.append(ft1 + ft2 + ft3 + ft4)
            self._auto_corr_deomposition.append([ft1, ft2, ft3, ft4])

    def evolve_single_step(self, evolve_dt):
        if self.j_oper2 is None:
            prev_bra_mpdm, prev_ket_mpdm = self.latest_mps
            prev_ket_mpdm2 = None
        else:
            (prev_bra_mpdm, prev_ket_mpdm), (prev_bra_mpdm, prev_ket_mpdm2) = self.latest_mps

        latest_ket_mpdm = prev_ket_mpdm.evolve(self.h_mpo, evolve_dt)
        latest_bra_mpdm = prev_bra_mpdm.evolve(self.h_mpo, evolve_dt)
        if self.j_oper2 is None:
            return BraKetPair(latest_bra_mpdm, latest_ket_mpdm, self.j_oper)
        else:
            latest_ket_mpdm2 = prev_ket_mpdm2.evolve(self.h_mpo, evolve_dt)
            return BraKetPair(latest_bra_mpdm, latest_ket_mpdm, self.j_oper), \
                   BraKetPair(latest_bra_mpdm, latest_ket_mpdm2, self.j_oper2)

    def stop_evolve_criteria(self):
        corr = self.auto_corr
        if len(corr) < 10:
            return False
        last_corr = corr[-10:]
        first_corr = corr[0]
        return np.abs(last_corr.mean()) < 1e-5 * np.abs(first_corr) and last_corr.std() < 1e-5 * np.abs(first_corr)

    @property
    def auto_corr(self) -> np.ndarray:
        """
        Correlation function :math:`C(t)`.

        :returns: 1-d numpy array containing the correlation function evaluated at each time step.
        """
        return np.array(self._auto_corr)

    @property
    def auto_corr_decomposition(self) -> np.ndarray:
        r"""
        Correlation function :math:`C(t)` decomposed into contributions from different parts
        of the current operator. Generally, the current operator can be split into two parts:
        current without phonon assistance and current with phonon assistance.
        For example, if the Holstein-Peierls model is considered:

        .. math::
            \hat H = \sum_{mn}  [\epsilon_{mn} + \sum_\lambda \hbar g_{mn\lambda} \omega_\lambda
            (b^\dagger_\lambda + b_\lambda) ] a^\dagger_m a_n
            + \sum_\lambda \hbar \omega_\lambda b^\dagger_\lambda  b_\lambda

        Then current operator without phonon assistance is defined as:

        .. math::
            \hat j_1 = \frac{e_0}{i\hbar} \sum_{mn} (R_m - R_n) \epsilon_{mn} a^\dagger_m a_n

        and the current operator with phonon assistance is defined as:

        .. math::
            \hat j_2 = \frac{e_0}{i\hbar} \sum_{mn} (R_m - R_n) \hbar g_{mn\lambda} \omega_\lambda
            (b^\dagger_\lambda + b_\lambda) a^\dagger_m a_n

        With :math:`\hat j = \hat j_1 + \hat j_2`, the correlation function can be
        decomposed into four parts:

        .. math::
            \begin{align}
            C(t) & = \langle \hat j(t) \hat j(0) \rangle \\
                 & = \langle ( \hat j_1(t) + \hat j_2(t) ) (\hat j_1(0) + \hat j_2(0) ) \rangle \\
                 & = \langle \hat j_1(t) \hat j_1(0) \rangle + \langle \hat j_1(t) \hat j_2(0) \rangle
                 + \langle \hat j_2(t) \hat j_1(0) \rangle + \langle \hat j_2(t) \hat j_2(0) \rangle
            \end{align}

        :return: :math:`n \times 4` array for the decomposed correlation function defined as above
            where :math:`n` is the number of time steps.
        """
        return np.array(self._auto_corr_deomposition)

    def get_dump_dict(self):
        dump_dict = dict()
        dump_dict["mol list"] = self.model.to_dict()
        dump_dict["temperature"] = self.temperature.as_au()
        dump_dict["time series"] = self.evolve_times
        dump_dict["auto correlation"] = self.auto_corr
        dump_dict["auto correlation decomposition"] = self.auto_corr_decomposition
        dump_dict["mobility"] = self.calc_mobility()[1]
        if self.properties is not None:
            for prop_str in self.properties.prop_res.keys():
                dump_dict[prop_str] = self.properties.prop_res[prop_str]

        return dump_dict

    def calc_mobility(self):
        time_series = self.evolve_times
        corr_real = self.auto_corr.real
        inte = scipy.integrate.trapz(corr_real, time_series)
        mobility_in_au = inte / self.temperature.as_au()
        mobility = mobility_in_au / mobility2au
        return mobility_in_au, mobility
