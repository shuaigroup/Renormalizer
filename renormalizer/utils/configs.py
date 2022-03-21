# -*- coding: utf-8 -*-
from enum import Enum
import logging

from renormalizer.utils.rk import RungeKutta, TaylorExpansion
import scipy.linalg
import numpy as np

logger = logging.getLogger(__name__)


class BondDimDistri(Enum):
    """
    Bond dimension distribution
    """
    # here the `#:` syntax is for sphinx documentation.
    #: uniform distribution.
    uniform = "uniform"
    #: Guassian distribution which peaks at the center.
    center_gauss = "center gaussian"


class CompressCriteria(Enum):
    """
    Criteria for compression.
    """
    #: Compress depending on pre-set threshold.
    #: States with singular value smaller than the threshold are discarded.
    threshold = "threshold"
    #: Compress depending on pre-set fixed bond dimension.
    fixed = "fixed"
    #: Compress combining ``threshold`` and ``fixed``. The bond dimension for the two criteria are both
    #: calculated and the smaller one is used.
    both = "both"


class OFS(Enum):
    """
    On the fly swapping criteria
    """
    # based on entanglement entropy
    ofs_s = "OFS-S"
    # a hybrid scheme
    ofs_ds = "OFS-D/S"
    # based on discarded weight
    ofs_d = "OFS-D"
    # debug mode. Dry run the code without performing the swapping
    ofs_debug = "OFS-Debug"


class CompressConfig:
    """MPS/MPO Compress Configuration.

    Parameters
    ----------
    criteria : `CompressCriteria`, optional
        The criteria for compression. Default is
        `CompressCriteria.threshold`.
    threshold : float, optional
        The threshold to keep states if ``criteria`` is set to
        `CompressCriteria.threshold` or `CompressCriteria.both`.
        Default is :math:`10^{-3}`.
    bonddim_distri : `BondDimDistri`, optional
        Bond dimension distribution if ``criteria`` is set to
        `CompressCriteria.fixed` or `CompressCriteria.both`.
        Default is `BondDimDistri.uniform`.
    max_bonddim : int, optional
        Maximum bond dimension ``M`` under various bond dimension distributions.
        Default is 32.
    vmethod : str, optional
        The algorithm used in variational compression. The default is ``2site``.

        - ``1site``:  optimize one site at a time during sweep.
        - ``2site``:  optimize two site at a time during sweep.

        Note
        ----
        It is recommended to use the 2site algorithm in variational optimization,
        though 1site algorithm is much cheaper. The reason is that 1site algorithm
        is easy to stuck into local minima while 2site algorithm is more robust. For
        complicated Hamiltonian, the problem is severer.  The renormalized basis
        selection rule used here to partly solve the local minimal problem is
        according to Chan. J.  Chem. Phys. 2004, 120, 3172. For efficiency, you
        could use 2site algorithm for several sweeps and then switch to 1site
        algorithm to converge the final mps.

    vprocedure : list, optional
        The procedure to do variational compression.
        The Default is

        .. code-block::

            if vmethod == "1site":
                vprocedure = [[max_bonddim,1.0],[max_bonddim,0.7],
                              [max_bonddim,0.5],[max_bonddim,0.3],
                              [max_bonddim,0.1]] + [[max_bonddim,0],]*10
            else:
                vprocedure = [[max_bonddim,0.5],[max_bonddim,0.3],
                              [max_bonddim,0.1]] + [[max_bonddim,0],]*10

    vrtol : float, optional
        The threshold of convergence in variational compression. Default is
        :math:`10^{-5}`.
    vguess_m : tuple(int), optional
        The bond dimension of ``compressed_mpo`` and ``compressed_mps`` to construct the initial guess
        ``compressed_mpo @ compressed_mps`` in `MatrixProduct.variational_compress`.
        The default is (5,5).
    dump_matrix_size : int or float, optional
        The total bytes threshold for dumping matrix into disks. Every local site matrix in the MPS/MPO
        with a size larger than the specified size will be moved from memory to the disk at ``dump_matrix_dir``.
        The matrix will be moved back to memory upon access.
        The dumped matrices on the disks will be removed once the MPS/MPO is garbage-collected.

        Note
        ----
        If ``dump_matrix_size`` is set and the program exits abnormally, it is possible that the dumped matrices
        are not deleted automatically. A common case when this can happen is that the program is killed by a signal.
        In such cases it is recommended to check and clean ``dump_matrix_dir`` after the exit of the program
        since the dumped matrices may take a lot of disk space.

    dump_matrix_dir : str, optional
        The directory to dump matrix when matrix is larger than ``dump_matrix_size``.

    ofs : `OFS`, optional
        Whether optimize the DOF ordering by OFS. The default value is ``None`` which means does not perform OFS.

    ofs_swap_jw : bool, optional
        Whether swap the Jordan-Wigner transformation ordering when OFS is enabled. Set to ``True``
        for and only for ab initio Hamiltonian constructed by the experimental
        ``renormalizer.model.h_qc.qc_model``. Default is ``False``.

    See Also
    --------
    CompressCriteria : Compression criteria
    renormalizer.mps.mp.MatrixProduct.variational_compress : Variational compression
    renormalizer.mps.Mpo.contract : compress ``mpo @ mps/mpdm/mpo``

    """
    def __init__(
        self,
        criteria: CompressCriteria = CompressCriteria.threshold,
        threshold: float = 1e-3,
        bonddim_distri: BondDimDistri = BondDimDistri.uniform,
        max_bonddim: int = 32,
        vmethod: str = "2site",
        vprocedure = None,
        vrtol = 1e-5,
        vguess_m = (5,5),
        dump_matrix_size = np.inf,
        dump_matrix_dir = "./",
        ofs: OFS = None,
        ofs_swap_jw: bool = False
    ):
        # two sets of criteria here: threshold and max_bonddimension
        # `criteria` is to determine which to use
        self.criteria: CompressCriteria = criteria
        self._threshold = None
        self.threshold = threshold
        self.bond_dim_distribution: BondDimDistri = bonddim_distri
        self.bond_dim_max_value = max_bonddim
        # not in arg list, but still useful in some cases
        self.bond_dim_min_value = 1
        # the length should be len(mps) + 1, the terminals are also counted. This is for accordance with mps.bond_dims
        self.max_dims: np.ndarray = None
        self.min_dims: np.ndarray = None

        # variational compression parameters
        self.vmethod = vmethod
        if vprocedure is None:
            if vmethod == "1site":
                vprocedure = [[max_bonddim,1.0],[max_bonddim,0.7],
                              [max_bonddim,0.5],[max_bonddim,0.3],
                              [max_bonddim,0.1]] + [[max_bonddim,0],]*10
            else:
                vprocedure = [[max_bonddim,0.5],[max_bonddim,0.3],
                              [max_bonddim,0.1]] + [[max_bonddim,0],]*10
        self.vprocedure = vprocedure
        self.vrtol = vrtol
        self.vguess_m = vguess_m

        self.dump_matrix_size = dump_matrix_size
        self.dump_matrix_dir = dump_matrix_dir

        self.ofs: OFS = ofs
        self.ofs_swap_jw: bool = ofs_swap_jw

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, v):
        if v <= 0:
            raise ValueError("non-positive threshold")
        elif v == 1:
            raise ValueError("1 is an ambiguous threshold")
        elif 1 < v:
            raise ValueError("Can't set threshold to be larger than 1")
        self._threshold = v

    def set_bonddim(self, length, max_value=None):
        if self.criteria is CompressCriteria.threshold:
            raise ValueError("compress config is using threshold criteria")
        if max_value is None:
            assert self.bond_dim_max_value is not None
            max_value = self.bond_dim_max_value
        else:
            self.bond_dim_max_value = max_value
        if max_value is None:
            raise ValueError("max value is not set")
        if self.bond_dim_distribution is BondDimDistri.uniform:
            self.max_dims = np.full(length, max_value, dtype=int)
        else:
            half_length = length // 2
            x = np.arange(- half_length, - half_length + length)
            sigma = half_length / np.sqrt(np.log(max_value / 3))
            seq = list(max_value * np.exp(-(x / sigma) ** 2))
            self.max_dims = np.int64(seq)
            assert not (self.max_dims == 0).any()
        self.min_dims = np.full(length, self.bond_dim_min_value, dtype=int)

    def _threshold_m_trunc(self, sigma: np.ndarray) -> int:
        assert 0 < self.threshold < 1
        # count how many sing vals < trunc
        normed_sigma = sigma / scipy.linalg.norm(sigma)
        return int(np.sum(normed_sigma > self.threshold))

    def _fixed_m_trunc(self, sigma: np.ndarray, idx: int, left: bool) -> int:
        assert self.max_dims is not None
        bond_idx = idx + 1 if left else idx
        return min(self.max_dims[bond_idx], len(sigma))

    def compute_m_trunc(self, sigma: np.ndarray, idx: int, left: bool) -> int:
        if self.criteria is CompressCriteria.threshold:
            trunc = self._threshold_m_trunc(sigma)
        elif self.criteria is CompressCriteria.fixed:
            trunc = self._fixed_m_trunc(sigma, idx, left)
        elif self.criteria is CompressCriteria.both:
            # use the smaller one
            trunc = min(
                self._threshold_m_trunc(sigma), self._fixed_m_trunc(sigma, idx, left)
            )
        else:
            assert False
        if self.min_dims is not None:
            bond_idx = idx if left else idx + 1
            min_trunc = min(self.min_dims[bond_idx], len(sigma))
            trunc = max(trunc, min_trunc)
        return trunc

    def update(self, other: "CompressConfig"):
        # use the stricter of the two
        if self.criteria != other.criteria:
            raise ValueError("Can't update configs with different standard")
        # look for minimum
        self.threshold = min(self.threshold, other.threshold)
        # look for maximum
        if self.max_dims is None:
            self.max_dims = other.max_dims
        elif other.max_dims is None:
            pass  # do nothing
        else:
            self.max_dims = np.maximum(self.max_dims, other.max_dims)

    def relax(self):
        # relax the two criteria simultaneously
        self.threshold = min(
            self.threshold * 3, 0.9
        )  # can't set to 1 which is ambiguous
        if self.max_dims is not None:
            self.max_dims = np.maximum(
                np.int64(self.max_dims * 0.8), np.full_like(self.max_dims, 2)
            )

    def copy(self) -> "CompressConfig":
        new = self.__class__.__new__(self.__class__)
        # shallow copies
        new.__dict__ = self.__dict__.copy()
        # deep copies
        if self.max_dims is not None:
            new.max_dims = self.max_dims.copy()
        if self.min_dims is not None:
            new.min_dims = self.min_dims.copy()
        return new

    @property
    def bonddim_should_set(self):
        return self.criteria is not CompressCriteria.threshold and self.max_dims is None

    def __str__(self):
        attrs = ["criteria", "threshold"]
        lines = []
        for attr in attrs:
            attr_value = getattr(self, attr)
            lines.append(f"\n{attr}: {attr_value}")
        return "".join(lines)


class OptimizeConfig:
    r""" DMRG ground state algorithm optimization configuration

    """

    def __init__(self, procedure=None):
        if procedure is None:
            self.procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]
        else:
            self.procedure = procedure
        self.method = "2site"
        # algo: arpack(Implicitly Restarted Lanczos Method)
        #       davidson(pyscf/davidson)
        #       primme (if installed)
        self.algo = "davidson"
        self.nroots = 1
        # the ground state energy convergence tolerance
        self.e_rtol = 1e-6
        self.e_atol = 1e-8
        # inverse = 1.0 or -1.0
        # -1.0 to get the largest eigenvalue
        self.inverse = 1.0


class EvolveMethod(Enum):
    """
    Time evolution methods.
    """
    # propagation and compression with RK45 propagator and time independent H
    prop_and_compress = "P&C"
    # propagation and compression with RK4 propagator and time dependent/independent H
    prop_and_compress_tdrk4 = "P&C TD RK4"
    # propagation and compression with RK propagator and time dependent/independent H
    prop_and_compress_tdrk = "P&C TD RK"
    # TDVP with projector splitting - one site
    tdvp_ps = "TDVP PS one-site"
    # TDVP with projector splitting - two site
    tdvp_ps2 = "TDVP PS two-site"
    # TDVP with variable mean field (VMF)
    tdvp_vmf = "TDVP Variable Mean Field"
    # TDVP with constant mean field (CMF) and matrix unfolding (MU) regularization
    tdvp_mu_cmf = "TDVP Matrix Unfolding Constant Mean Field"
    # TDVP with variable mean field (VMF) and matrix unfolding (MU) regularization
    tdvp_mu_vmf = "TDVP Matrix Unfolding Variable Mean Field"


def parse_memory_limit(x) -> float:
    if x is None:
        return float("inf")
    try:
        return float(x)
    except (TypeError, ValueError):
        pass
    try:
        x_str = str(x)
        num, unit = x_str.split()
        unit = unit.lower()
        mapping = {"kb": 2 ** 10, "mb": 2 ** 20, "gb": 2 ** 30}
        return float(num) * mapping[unit]
    except:
        # might error when converting to str, but the message is clear enough.
        raise ValueError(f"invalid input for memory: {x}")


class EvolveConfig:
    def __init__(
        self,
        method: EvolveMethod = EvolveMethod.prop_and_compress,
        adaptive=False,
        guess_dt=1e-1,
        adaptive_rtol=5e-4,
        taylor_order:int = None,
        rk_solver="C_RK4",
        reg_epsilon=1e-10,
        ivp_rtol=1e-5,
        ivp_atol=1e-8,
        force_ovlp=True
    ):

        self.method = method
        self.adaptive = adaptive
        self.rk_config = RungeKutta(rk_solver)
        if taylor_order is None:
            if adaptive:
                taylor_order = 5  
            else:
                taylor_order = 4
        self.taylor_config = TaylorExpansion(taylor_order)

        self.guess_dt: complex = guess_dt  # a guess of initial adaptive time step
        self.adaptive_rtol = adaptive_rtol

        self.tdvp_cmf_midpoint = True
        self.tdvp_cmf_c_trapz = False
        # regularization parameter in tdvp_mu or tdvp_std method
        self.reg_epsilon: float = reg_epsilon
        # scipy.ivp rtol and atol
        self.ivp_rtol: float = ivp_rtol
        self.ivp_atol: float = ivp_atol
        # the EOM has already considered the non-orthogonality of the left and right
        # renormalized basis, see arXiv:1907.12044
        self.force_ovlp: bool = force_ovlp
        # auto switch between mu_vmf and vmf for a higher efficiency
        self.vmf_auto_switch: bool = True

    @property
    def is_tdvp(self):
        return self.method not in [EvolveMethod.prop_and_compress,
                EvolveMethod.prop_and_compress_tdrk4,
                EvolveMethod.prop_and_compress_tdrk]

    def check_valid_dt(self, evolve_dt: complex):
        info_str = f"in config: {self.guess_dt}, in arg: {evolve_dt}"

        if np.iscomplex(evolve_dt) ^ np.iscomplex(self.guess_dt):
            raise ValueError("real and imag not compatible. " + info_str)

        if (np.iscomplex(evolve_dt) and evolve_dt.imag * self.guess_dt.imag < 0) or \
                (not np.iscomplex(evolve_dt) and evolve_dt * self.guess_dt < 0):
            raise ValueError("evolve into wrong direction. " + info_str)


    def copy(self):
        new = self.__class__.__new__(self.__class__)
        new.__dict__ = self.__dict__.copy()
        return new

    def __str__(self):
        attrs = list(self.__dict__.keys())
        lines = []
        for attr in attrs:
            attr_value = getattr(self, attr)
            lines.append(f"\n{attr}: {attr_value}")
        return "".join(lines)
