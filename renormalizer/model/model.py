# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import logging
from typing import List, Union, Dict, Callable

import numpy as np

from renormalizer.model.basis import BasisSet, BasisSimpleElectron, BasisMultiElectronVac, BasisHalfSpin, BasisSHO
from renormalizer.model.mol import Mol, Phonon
from renormalizer.model.op import Op
from renormalizer.utils import Quantity, cached_property


logger = logging.getLogger(__name__)

class Model:
    r"""
    The most general model that supports any Hamiltonian in sum-of-product form.
    Base class for :class:`HolsteinModel` and :class:`SpinBosonModel`.

    Parameters
    ==========
    basis : :class:`list` of :class:`~renormalizer.model.basis.BasisSet`
        Local basis for each site of the MPS. The order determines the
        DoF order in the MPS.
    ham_terms : :class:`list` of :class:`~renormalizer.model.Op`
        Terms of the system Hamiltonian in sum-of-product form.
        Identities can be omitted in the operators.
        All terms must be included, without assuming Hermitian or something else.
    dipole : dict
        Contains the transition dipole matrix element. The key is the dof name.
    output_ordering : :class:`list` of :class:`~renormalizer.model.basis.BasisSet`
        The ordering of the local basis for output. Default is the same with ``basis``.
    """
    def __init__(self, basis: List[BasisSet], ham_terms: List[Op], dipole: Dict = None, output_ordering: List[BasisSet]=None):
        if not isinstance(basis, list) or len(basis) == 0:
            raise TypeError("Basis should be a non-empty list")
        if not isinstance(basis[0], BasisSet):
            raise TypeError("Elements of the basis list should be of type BasisSet")
        all_dof_list = []
        for local_basis in basis:
                all_dof_list.extend(local_basis.dofs)
        if len(all_dof_list) != len(set(all_dof_list)):
            raise ValueError("Duplicate DoF definition found in the basis list.")
        self.basis: List[BasisSet] = basis
        if output_ordering is None:
            self.output_ordering = self.basis
        else:
            self.output_ordering = output_ordering

        # alias
        self.dof_to_siteidx = self.order = {}
        self.dof_to_basis = {}
        for siteidx, ba in enumerate(basis):
            for dof_name in ba.dofs:
                self.dof_to_siteidx[dof_name] = siteidx
                self.dof_to_basis[dof_name] = ba

        self.ham_terms: List[Op] = self.check_operator_terms(ham_terms)
        # array(n_e, n_e)
        self.dipole = dipole
        # reusable mpos for the system
        self.mpos = dict()
        # physical bond dimension.
        self.pbond_list = [local_basis.nbas for local_basis in self.basis]

    def check_operator_terms(self, terms: List[Op]):
        """
        Check and clean operator terms in the input ``terms``.
        Errors will be raised if the type of operator is not :class:`Op`
        or the operator contains DoF not defined in ``self.basis``.
        Operators with factor = 0 are discarded.

        Parameters
        ----------
        terms : :class:`list` of :class:`~renormalizer.model.Op`
            The terms to check.

        Returns
        -------
        new_terms: :class:`list` of :class:`Op`
            Operator list with 0-factor terms discarded.
        """
        # terms to return
        new_terms = []
        dofs = set(self.dofs)
        for term_op in terms:
            if not isinstance(term_op, Op):
                raise ValueError(f"Expected Op in terms. Got {term_op}")
            for name in term_op.dofs:
                if name not in dofs:
                    raise ValueError(f"{term_op} contains DoF not in the basis.")
            # discard terms with 0 factor
            if term_op.factor == 0:
                continue
            new_terms.append(term_op)
        return new_terms

    def _enumerate_dof(self, criteria=lambda x: True):
        # enumerate DoFs and filter according to criteria.
        dofs = []
        for local_basis in self.output_ordering:
            if criteria(local_basis):
                dofs.extend(local_basis.dofs)
        return dofs

    @cached_property
    def dofs(self) -> List:
        """
        :class:`list` of DoF names.
        """
        return self._enumerate_dof()
    
    @cached_property
    def nsite(self) -> int:
        """
        Number of sites in the MPS/MPO to be constructed.
        Length of ``self.basis``.
        """
        return len(self.basis)

    @cached_property
    def e_dofs(self) -> List:
        """
        :class:`list` of electronic DoF names.
        """
        return self._enumerate_dof(lambda basis: basis.is_electron)
    
    @cached_property
    def v_dofs(self) -> List:
        """
        :class:`list` of vibrational DoF names.
        """
        return self._enumerate_dof(lambda basis: basis.is_phonon)

    @cached_property
    def n_dofs(self) -> int:
        """
        Number of total DoFs.
        """
        return len(self.dofs)

    @cached_property
    def n_edofs(self) -> int:
        """
        Number of total electronic DoFs.
        """
        return len(self.e_dofs)
    
    @cached_property
    def n_vdofs(self) -> int:
        """
        Number of total vibrational DoFs.
        """
        return len(self.v_dofs)

    def get_mpos(self, key: str, fun: Callable):
        r"""
        Get MPOs related to the model, such as MPOs to calculate
        electronic occupations :math:`{a^\dagger_i a_i}`.
        The purpose of the function is to avoid repeated MPO construction.

        Parameters
        ----------
        key : str
            Name of the MPOs. In principle other hashable types are also OK.
        fun : callable
            The function to generate MPOs if the MPOs have not been
            constructed before. The function should accept only one argument
            which is the model and return a :class:`list` of
            :class:`~renormalizer.mps.Mpo`.

        Returns
        -------
        mpos : list of :class:`~renormalizer.mps.Mpo`
            The required MPOs.
        """
        if key not in self.mpos:
            mpos = fun(self)
            self.mpos[key] = mpos
        else:
            mpos = self.mpos[key]
        return mpos

    def copy(self):
        # copy basis because it is mutable with OFS
        model =  Model(self.basis.copy(), self.ham_terms, self.dipole, self.output_ordering)
        # this is a shallow copy, in order to avoid infinite recursion
        model.mpos = self.mpos.copy()
        return model

    def to_dict(self) -> Dict:
        """
        Convert the object into a dict that contains only objects of
        Python primitive types or NumPy types.
        This is primarily for dump purposes.

        Returns
        -------
        info_dict : dict
            The information of the model in a dict.
        """
        info_dict = {}
        # todo: dump basis
        info_dict["Hamiltonian"] = [op.to_tuple() for op in self.ham_terms]
        info_dict["dipole"] = self.dipole
        return info_dict


class HolsteinModel(Model):
    r"""
    Interface for convenient Holstein model construction.
    The Hamiltonian of the Holstein model:

    .. math::
        \hat H = \sum_{ij} J_{ij} a^\dagger_i a_j + \sum_{i\lambda} \omega_{i\lambda} b^\dagger_{i\lambda} b_{i\lambda}
        + \sum_{i\lambda} g_{i\lambda} \omega_{i\lambda} a^\dagger_i a_i (b^\dagger_{i\lambda} + b_{i\lambda})

    Parameters
    ==========
    mol_list : :class:`list` of :class:`~renormalizer.model.Mol`
        Information for the molecules contains in the system.
        See the :class:`~renormalizer.model.Mol` class for more details.
    j_matrix : :class:`np.ndarray` or :class:`~renormalizer.utils.Quantity`.
        :math:`J_{ij}` in the Holstein Hamiltonian. When :class:`Quantity` is used as input, the system is taken to
        have homogeneous nearest-neighbour interaction
        :math:`J_{ij}=J \delta_{i, j+1} + J \delta_{i, j-1}`.
        For the boundary condition, see the ``periodic`` option.
    scheme : int
        The scheme of the basis for the model. Historically four numbers are permitted: 1, 2, 3, 4.
        Now 1, 2 and 3 are equivalent, and the bases are arranged as:

        .. math::
            [\rm{e}_{0}, \rm{ph}_{0,0}, \rm{ph}_{0,1}, \cdots, \rm{e}_{1}, \rm{ph}_{1,0}, \rm{ph}_{1,1}, \cdots ]

        And when ``scheme`` is set to 4, all electronic DoF is contained in one
        :class:`~renormalizer.model.basis.BasisSet`
        using :class:`~renormalizer.model.basis.BasisMultiElectronVac` and the bases are arranged as:

        .. math::
            [\rm{ph}_{0,0},  \rm{ph}_{0,1}, \cdots, \rm{ph}_{n/2, 0}, \rm{ph}_{n/2, 1}, \cdots, \rm{e}_{0, 1, \cdots, n}
            \rm{ph}_{n/2+1, 0}, \rm{ph}_{n/2+1, 1}, \cdots]

    periodic : bool
        Whether use periodical boundary condition when constructing ``j_matrix``
        from :class:`~renormalizer.utils.Quantity`. Default is ``False``.
    """

    def __init__(self,  mol_list: List[Mol], j_matrix: Union[Quantity, np.ndarray], scheme: int = 2, periodic: bool = False):
        # construct the electronic coupling matrix

        mol_num = len(mol_list)
        self.mol_list = mol_list

        if isinstance(j_matrix, Quantity):
            j_matrix = construct_j_matrix(mol_num, j_matrix, periodic)
        else:
            if periodic:
                assert j_matrix[0][-1] != 0 and j_matrix[-1][0] != 0
            assert j_matrix.shape[0] == mol_num

        self.j_matrix = j_matrix
        self.scheme = scheme

        basis = []
        ham = []

        if scheme < 4:
            for imol, mol in enumerate(mol_list):
                basis.append(BasisSimpleElectron(imol))
                for iph, ph in enumerate(mol.ph_list):
                    basis.append(BasisSHO((imol, iph), ph.omega[0], ph.n_phys_dim))

        elif scheme == 4:

            n_left_mol = mol_num // 2

            n_left_ph = 0
            for imol, mol in enumerate(mol_list):
                for iph, ph in enumerate(mol.ph_list):
                    if imol < n_left_mol:
                        n_left_ph += 1
                    basis.append(BasisSHO((imol, iph), ph.omega[0], ph.n_phys_dim))

            basis.insert(n_left_ph, BasisMultiElectronVac(list(range(len(mol_list)))))

        else:
            raise ValueError(f"invalid model.scheme: {scheme}")

        # model

        # electronic term
        for imol in range(mol_num):
            for jmol in range(mol_num):
                if imol == jmol:
                    factor =  mol_list[imol].elocalex + mol_list[imol].e0
                else:
                    factor = j_matrix[imol, jmol]
                ham_term = Op(r"a^\dagger a", [imol, jmol], factor)
                ham.append(ham_term)
        # vibration part
        for imol, mol in enumerate(mol_list):
            for iph, ph in enumerate(mol.ph_list):

                ham.extend([
                    Op("p^2", (imol, iph), 0.5),
                    Op("x^2", (imol, iph), 0.5 * ph.omega[0] ** 2)
                ])

        # vibration potential part
        for imol, mol in enumerate(mol_list):
            for iph, ph in enumerate(mol.ph_list):
                if np.allclose(ph.omega[0], ph.omega[1]):
                    ham.append(
                        Op(r"a^\dagger a", imol) * Op("x", (imol,iph)) * (-ph.omega[1] ** 2 * ph.dis[1])
                    )
                else:
                    ham.extend([
                        Op(r"a^\dagger a", imol) * Op("x^2", (imol, iph)) * (0.5 * (ph.omega[1] ** 2 - ph.omega[0] ** 2)),
                        Op(r"a^\dagger a", imol) * Op("x", (imol, iph)) * (-ph.omega[1] ** 2 * ph.dis[1]),
                    ])


        dipole = {}
        for imol, mol in enumerate(mol_list):
            dipole[imol] = mol.dipole

        super().__init__(basis, ham, dipole=dipole)
        self.mol_num = self.n_edofs

    def switch_scheme(self, scheme: int) -> "HolsteinModel":
        """
        Switch the scheme of the current model.

        Parameters
        ----------
        scheme : int
            The target scheme.

        Returns
        -------
        new_model : HolsteinModel
            The new model with the specified scheme.
        """
        return HolsteinModel(self.mol_list, self.j_matrix, scheme)

    @property
    def gs_zpe(self) -> float:
        r"""
        Ground state zero-point energy :math:`\sum_{i, j} \frac{1}{2}\omega_{i, j}`.
        """
        return sum([mol.gs_zpe for mol in self.mol_list])

    @property
    def j_constant(self):
        """Extract electronic coupling constant from ``self.j_matrix``.
        Useful in transport model when :math:`J` is a constant..
        If J is actually not a constant, a value error will be raised.

        Returns
        -------
        j constant: float
            J constant extracted from ``self.j_matrix``.
        """
        j_set = set(self.j_matrix.ravel())
        if len(j_set) == 0:
            return j_set.pop()
        elif len(j_set) == 2 and 0 in j_set:
            j_set.remove(0)
            return j_set.pop()
        else:
            raise ValueError("J is not constant")

    def copy(self):
        model = HolsteinModel(self.mol_list, self.j_matrix, self.scheme)
        model.mpos = self.mpos.copy()
        return model

    def __getitem__(self, item):
        return self.mol_list[item]

    def __iter__(self):
        return iter(self.mol_list)

    def __len__(self):
        return len(self.mol_list)


class SpinBosonModel(Model):
    r"""
    Spin-Boson model

        .. math::
            \hat{H} = \epsilon \sigma_z + \Delta \sigma_x
                + \frac{1}{2} \sum_i(p_i^2+\omega^2_i q_i^2)
                + \sigma_z \sum_i c_i q_i

    """
    def __init__(self, epsilon: Quantity, delta: Quantity, ph_list: List[Phonon], dipole: float=None):

        self.epsilon = epsilon.as_au()
        self.delta = delta.as_au()
        self.ph_list = ph_list

        basis = [BasisHalfSpin("spin")]
        for iph, ph in enumerate(ph_list):
            basis.append(BasisSHO(iph, ph.omega[0], ph.n_phys_dim))

        # spin
        ham = [Op(r"sigma_z", "spin", self.epsilon), Op("sigma_x", "spin", self.delta)]
        # vibration energy and potential
        for iph, ph in enumerate(ph_list):
            assert ph.is_simple
            ham.extend([Op("p^2", iph, 0.5), Op("x^2", iph, 0.5 * ph.omega[0] ** 2)])
            ham.append(Op("sigma_z", "spin") * Op("x", iph) *(-ph.omega[1] ** 2 * ph.dis[1]))
        if dipole is None:
            dipole = 0
        super().__init__(basis, ham, dipole={"spin": dipole})


class TI1DModel(Model):
    r"""
    Translational invariant one dimensional model with periodic boundary condition.
    The Hamiltonian should take the form:

    .. math::
        \hat H = \sum_i(\hat h_i + \sum_j \hat h_{i,j})

    where :math:`\hat h_i` is the local Hamiltonian acting on one single unit cell
    and :math:`\hat h_{i,j}` represents the :math:`j` th interaction between the :math:`i` th cell
    and other unit cells.

    Yet doesn't support setting transition dipoles.

    Parameters
    ==========
    basis : :class:`list` of :class:`~renormalizer.model.basis.BasisSet`
        Local basis of each site for a single unit cell of the system.
        The full basis set is constructed by repeating the basis ``ncell`` times.
        To distinguish between different DoFs at different unit cells,
        the DoF names in the unit cell are transformed to
        a two-element-tuple of the form ``("cell0", dof)``, where ``dof`` is the original DoF name
        and the ``"cell0"`` indicates the cell ID.
    local_ham_terms : :class:`list` of :class:`~renormalizer.model.Op`
        Terms of the system local Hamiltonian :math:`\hat h_i` in sum-of-product form.
        DoF names should be consistent with the ``basis`` argument.
    nonlocal_ham_terms : :class:`list` of :class:`~renormalizer.model.Op`
        Terms of system nonlocal Hamiltonian :math:`\hat h_{i,j}`.
        To indicate the IDs of the unit cells that are involved in the nonlocal interaction,
        the DoF name in ``basis`` should be transformed to a two-element-tuple, in which
        the first element is an integer indicating its distance from :math:`i`,
        and the second element is the original DoF name.
        For example, if one unit cell contains one electron DoF with name ``e``,
        a nearest neighbour hopping interaction
        should take the form ``Op(r"a^\dagger a", [(0, "e"), (1, "e")])``
        (with its Hermite conjugation being another term).
        The definition is not unique in that ``Op(r"a^\dagger a", [(1, "e"), (2, "e")])``
        produces equivalent output.
    ncell : int
        Number of unit cells in the system.
    """
    def __init__(self, basis: List[BasisSet], local_ham_terms: List[Op], nonlocal_ham_terms: List[Op], ncell: int):
        # construct basis for the full model
        full_basis = []
        for i in range(ncell):
            for local_basis in basis:
                new_dofs = [(f"cell{i}", dof) for dof in local_basis.dofs]
                if local_basis.multi_dof:
                    new_basis = local_basis.copy(new_dofs)
                else:
                    new_basis = local_basis.copy(new_dofs[0])
                full_basis.append(new_basis)
        # construct Hamiltonian for the full model
        full_ham_terms = []
        for i in range(ncell):
            for old_op in local_ham_terms:
                new_dofs = [(f"cell{i}", dof) for dof in old_op.dofs]
                new_op = Op(old_op.symbol, new_dofs, old_op.factor, old_op.qn_list)
                full_ham_terms.append(new_op)
            for old_op in nonlocal_ham_terms:
                new_dofs = []
                for old_dof in old_op.dofs:
                    assert isinstance(old_dof, tuple) and len(old_dof) == 2 and isinstance(old_dof[0], int)
                    # take care of periodic boundary condition
                    new_cell_id = (i + old_dof[0]) % ncell
                    new_dofs.append((f"cell{new_cell_id}", old_dof[1]))
                new_op = Op(old_op.symbol, new_dofs, old_op.factor, old_op.qn_list)
                full_ham_terms.append(new_op)
        super().__init__(full_basis, full_ham_terms)


def construct_j_matrix(mol_num, j_constant, periodic):
    # nearest neighbour interaction
    j_constant_au = j_constant.as_au()
    j_list = np.ones(mol_num - 1) * j_constant_au
    j_matrix = np.diag(j_list, k=-1) + np.diag(j_list, k=1)
    if periodic:
        j_matrix[-1, 0] = j_matrix[0, -1] = j_constant_au
    return j_matrix


def load_from_dict(param, scheme, lam: bool):
    temperature = Quantity(*param["temperature"])
    ph_list = [
        Phonon.simplest_phonon(
            Quantity(*omega), Quantity(*displacement), temperature=temperature, lam=lam
        )
        for omega, displacement in param["ph modes"]
    ]
    j_constant = Quantity(*param["j constant"])
    model = HolsteinModel([Mol(Quantity(0), ph_list)] * param["mol num"], j_constant, scheme)
    return model, temperature
