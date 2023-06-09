# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

from typing import List, Union, Dict, Callable

import numpy as np

from renormalizer.model.basis import BasisSet, BasisSimpleElectron, BasisMultiElectronVac, BasisHalfSpin, BasisSHO
from renormalizer.model.mol import Mol, Phonon
from renormalizer.model.op import Op
from renormalizer.utils import Quantity, cached_property


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
    """
    def __init__(self, basis: List[BasisSet], ham_terms: List[Op], 
            dipole: Dict = None, 
            para: Dict = None):

        self.basis: List[BasisSet] = basis
        # alias
        self.dof_to_siteidx = self.order = {}
        for siteidx, ba in enumerate(basis):
            for dof_name in ba.dofs:
                self.dof_to_siteidx[dof_name] = siteidx

        self.ham_terms: List[Op] = self.check_operator_terms(ham_terms)
        # array(n_e, n_e)
        self.dipole = dipole
        # reusable mpos for the system
        self.mpos = dict()
        # physical bond dimension.
        self.pbond_list = [bas.nbas for bas in self.basis]
        
        if para is None:
            self.para = {}
        else:
            self.para = para

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
                raise ValueError("Expected Op in terms.")
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
        for basis in self.basis:
            if criteria(basis):
                dofs.extend(basis.dofs)
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

    def __init__(self,  mol_list: List[Mol], j_matrix: Union[Quantity, np.ndarray, None], scheme: int = 2, periodic: bool = False):
        # construct the electronic coupling matrix

        mol_num = len(mol_list)
        self.mol_list = mol_list

        if j_matrix is None:
            # spin-boson model
            assert len(mol_list) == 1
            j_matrix = Quantity(0)

        if isinstance(j_matrix, Quantity):
            j_matrix = construct_j_matrix(mol_num, j_matrix, periodic)
        else:
            if periodic:
                assert j_matrix[0][-1] != 0 and j_matrix[-1][0] != 0
            assert j_matrix.shape[0] == mol_num

        self.j_matrix = j_matrix
        self.scheme = scheme
        self.periodic = periodic

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
                assert np.allclose(np.array(ph.force3rd), [0.0, 0.0])

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
