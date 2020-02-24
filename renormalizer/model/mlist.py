# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

from collections import OrderedDict
from typing import List, Union, Dict
from enum import Enum

import numpy as np

from renormalizer.model.ephtable import EphTable
from renormalizer.model.mol import Mol, Phonon
from renormalizer.utils import Quantity, Op
from renormalizer.utils.utils import cast_float
from renormalizer.utils import basis as ba

class ModelTranslator(Enum):
    """
    Available Translator from user input model to renormalizer's internal formats 
    """
    # from MolList scheme1/2/3
    Holstein_model_scheme123 = "Holstein Model Scheme123 (MolList)"
    # from MolList scheme4
    Holstein_model_scheme4 = "Holstein Model Scheme4 (MolList)"
    # from MolList sbm
    sbm = "Spin Boson Model (MolList)"
    # from MolList2 or MolList augmented with MolList2 parameters
    vibronic_model = "Vibronic Model (MolList2)"
    # from MolList2 or MolList augmented with MolList2 parameters
    general_model = "General Model (MolList2)"


class MolList2:
    def __init__(self, order:Dict, basis:List, model:Dict, model_translator: ModelTranslator, dipole:Dict=None):
        r"""
        User-defined model
        
        Args:
            order (Dict): order of degrees of freedom.
            basis (List[class:`~renormalizer.utils.basis]): local basis of each DoF
            model (Dict): model of the system or any operator of the system, two
            formats are supported: 'vibronic type' or 'general type'. All terms
            must be included, without assuming hermitian or something else.
            model_translator(:class:`~renormalizer.model.ModelTranslator`): Translator from user input model to renormalizer's internal formats
            dipole (Dict): contains the transition dipole matrix element
        
        Note:
            the key of order starts from "v" or "e" for vibrational DoF or electronic DoF respectively. 
            after the linker '_' combined with an index. The rule is that
            the index of both 'e' and 'v' starts from 0,1,2... and the 
            properties such as `e_occupations` are reported with such order.
            the value of order is the position of the specific DoF, starting from 0,2,...,nsite-1
            For cases that all electronic DoFs gathered in a single site,
            the value of each DoF should be the same.
            for example: MolList scheme1/2/3 order = {"e_0":0, "v_0":1, "v_1":2,
                "e_1":3, "v_2":4, "v_3":5}
                         MolList scheme4 order ={"e_0":0, "v_0":1, "v_1":2,
                         "e_1":0, "v_2":3, "v_3":4}
        
            The order of basis corresponds to the site, each element is a Basis
            class, refer to `~renormalizer.utils.basis`
            for example: basis = [b0,b1,b2,b3] 

            two formats are supported for model:
            'vibronic type': each key is a tuple of electronic DoFs represents
            a^\dagger_i a_j or the key is "I" represents the pure vibrational
            terms, the value is a dict. 
            The sub-key of the dict has two types, one is 'J' with value (float or complex) for pure electronic coupling,
            one is tuple of vibrational DoF with a list as value. Inside the
            list is sevaral tuples, each tuple is a operator term. the last one
            of the tuple is factor of the term, the others represents a local
            operator (refer to `~renormalizer.utils.Op`) on each Dof in the
            sub-key (one-to-one map between the sub-key and tuple).
            The model_translator is ModelTranslator.vibronic_model
            for example:
            {"I": {("v_1"):[(Op,factor),]}, ("e_i","e_j") : {"J":factor, ("v_0",):[(Op, factor), (Op, factor)],
            ("v_1","v_2"):[(Op1, Op2, factor), (Op1, Op2, factor)]...}...}
            
            'general type': each key is a tuple of DoFs,
            each value is list. Inside the list, each element is a tuple, the
            last one is the factor of each operator term, the others are local
            operator of the operator term.
            The model_translator is ModelTranslator.general_model
            for example: 
            {("e_i","e_j") : [(Op1, Op2, factor)], ("e_i", "v_0",):[(Op1, Op2,
            factor), (Op1, Op2, factor)], ("v_1","v_2"):[(Op1, Op2, factor), (Op1, Op2, factor)]...}

            dipole contains transtion dipole matrix elements between the
            electronic DoFs. For simple electron case, dipole = {("e_0",):tdm1,
            ("e_1",):tdm2}, the key has 1 e_dof. For multi electron case, dipole =
            {("e_0","e_1"):tdm1, ("e_1","e_0"):tdm1}, the key has 2 e_dofs
            represents the transition.
        """
        self.order = order
        self.basis = basis
        self.model = model
        self.model_translator = model_translator
        
        # array(n_e, n_e)
        self.dipole = dipole
        # map: to be compatible with MolList {(imol, iph):"v_n"}
        self.map = None
        # reusable mpos for the system
        self.mpos = dict()
            
    @property
    def multi_electron(self):
        for b in self.basis:
            if isinstance(b, ba.BasisMultiElectron):
                return True
        return False

    @property
    def pbond_list(self):
        return [bas.nbas for bas in self.basis]
    
    def rewrite_model(self, model, model_translator):
        return self.__class__(self.order, self.basis, model, model_translator,
                self.dipole)   

    @property
    def dofs(self):
        # If items(), keys(), values(),  iteritems(), iterkeys(), and
        # itervalues() are called with no intervening modifications to the
        # dictionary, the lists will directly correspond.
        
        return list(self.order.keys())
    
    @property
    def j_matrix(self):
        J = np.zeros((self.e_nsite, self.e_nsite))
        for i, idof in enumerate(self.e_dofs):
            for j, jdof in enumerate(self.e_dofs):
                if self.model_translator == ModelTranslator.vibronic_model:
                    if (f"{idof}", f"{jdof}") in self.model.keys():
                        J[i,j] = self.model[(f"{idof}", f"{jdof}")]["J"]
                elif self.model_translator == ModelTranslator.general_model:
                    if (f"{idof}", f"{jdof}") in self.model.keys():
                        for term in self.model[(f"{idof}", f"{jdof}")]:
                            if term[0].symbol == r"a^\dagger" and term[1].symbol == "a":
                                J[i,j] = term[-1]
                                break
                    
                    if (f"{jdof}", f"{idof}") in self.model.keys():
                        for term in self.model[(f"{jdof}", f"{idof}")]:
                            if term[0].symbol == r"a" and term[1].symbol == r"a^\dagger":
                                J[i,j] = term[-1]
                                break
                else:
                    raise ValueError("j_matrix doesn't support {self.model_translator}")
        assert np.allclose(J, J.T)            
        return J
    
    @property
    def nsite(self):
        return len(self.basis)

    @property
    def e_dofs(self):
        dofs = []
        for key in self.order.keys():
            if key.split("_")[0] == "e":
                dofs.append(int(key.split("_")[1]))
        assert sorted(dofs) == list(range(len(dofs)))
        return [f"e_{i}" for i in range(len(dofs))]
    
    @property
    def v_dofs(self):
        dofs = []
        for key in self.order.keys():
            if key.split("_")[0] == "v":
                dofs.append(int(key.split("_")[1]))
        assert sorted(dofs) == list(range(len(dofs)))
        return [f"v_{i}" for i in range(len(dofs))]
   
    @property
    def e_nsite(self):
        return len(self.e_dofs)
    
    @property
    def v_nsite(self):
        return len(self.v_dofs)

    
    @property
    def pure_dmrg(self):
        return True
    
    def to_dict(self):
        info_dict = OrderedDict()
        info_dict["order"] = self.order
        info_dict["model"] = self.model
        info_dict["model_translator"] = self.model_translator
        return info_dict

    @classmethod
    def MolList_to_MolList2(cls, mol_list, formula="vibronic"):
        """
        switch from MolList to MolList2
        """
        
        order = {}
        basis = []
        model = {}
        mapping = {}
        
        def e_idx(idx):
            if mol_list.scheme < 4:
                return idx
            elif mol_list.scheme == 4:
                # add the gs state
                return idx + 1

        if mol_list.scheme < 4:
            idx = 0
            nv = 0
            for imol, mol in enumerate(mol_list):
                order[f"e_{e_idx(imol)}"] = idx
                if np.allclose(mol.tunnel, 0):
                    basis.append(ba.BasisSimpleElectron())
                else:
                    basis.append(ba.BasisHalfSpin())
                idx += 1
                for iph, ph in enumerate(mol.dmrg_phs):
                    order[f"v_{nv}"] = idx
                    mapping[(imol, iph)] = f"v_{nv}"
                    basis.append(ba.BasisSHO(ph.omega[0], ph.n_phys_dim))
                    idx += 1
                    nv += 1

        elif mol_list.scheme == 4:
            
            n_left_mol = mol_list.mol_num // 2
            
            idx = 0
            n_left_ph = 0
            nv = 0
            for imol, mol in enumerate(mol_list):
                for iph, ph in enumerate(mol.dmrg_phs):
                    if imol < n_left_mol:
                        order[f"v_{nv}"] = idx
                        n_left_ph += 1
                    else:
                        order[f"v_{nv}"] = idx+1
                    
                    basis.append(ba.BasisSHO(ph.omega[0], ph.n_phys_dim))
                    mapping[(imol, iph)] = f"v_{nv}"

                    nv += 1
                    idx += 1
            
            for imol in range(mol_list.mol_num):
                order[f"e_{e_idx(imol)}"] = n_left_ph
            # the gs state
            order["e_0"] = n_left_ph
            basis.insert(n_left_ph,
                         ba.BasisMultiElectron(mol_list.mol_num + 1, [0, ] + [1, ] * mol_list.mol_num))

        else:
            raise ValueError(f"invalid mol_list.scheme: {mol_list.scheme}")
        

        # model
        if formula == "vibronic":
            # electronic term
            for imol in range(mol_list.mol_num):
                for jmol in range(mol_list.mol_num):
                    if imol == jmol:
                        model[(f"e_{e_idx(imol)}", f"e_{e_idx(jmol)}")] = {"J": mol_list[imol].elocalex + mol_list[imol].dmrg_e0}
                    else:
                        model[(f"e_{e_idx(imol)}", f"e_{e_idx(jmol)}")] = {"J": mol_list.j_matrix[imol, jmol]}
            
            # vibration part
            model["I"] = {}
            for imol, mol in enumerate(mol_list):
                for iph, ph in enumerate(mol.dmrg_phs):
                    assert np.allclose(np.array(ph.force3rd), [0.0, 0.0])

                    model["I"][(mapping[(imol, iph)],)] = [(Op("p^2", 0), 0.5),
                            (Op("x^2", 0), 0.5*ph.omega[0]**2)]

            # vibration potential part
            for imol, mol in enumerate(mol_list):
                for iph, ph in enumerate(mol.dmrg_phs):
                    if np.allclose(ph.omega[0], ph.omega[1]):
                        model[(f"e_{e_idx(imol)}", f"e_{e_idx(imol)}")][(mapping[(imol,iph)],)] \
                            = [(Op("x", 0), -ph.omega[1]**2*ph.dis[1])]
                    
                    else:
                        model[(f"e_{e_idx(imol)}", f"e_{e_idx(imol)}")][(mapping[(imol,iph)],)] \
                            = [
                                (Op("x^2", 0), 0.5*(ph.omega[1]**2-ph.omega[0]**2)),
                                (Op("x", 0), -ph.omega[1]**2*ph.dis[1]),
                                ]

            model_translator = ModelTranslator.vibronic_model
        
        elif formula == "general":
            # electronic term
            for imol in range(mol_list.mol_num):
                for jmol in range(mol_list.mol_num):
                    if imol == jmol:
                        model[(f"e_{e_idx(imol)}",)] = \
                        [(Op(r"a^\dagger a", 0),
                            mol_list[imol].elocalex + mol_list[imol].dmrg_e0)]
                    else:
                        model[(f"e_{e_idx(imol)}", f"e_{e_idx(jmol)}")] = \
                            [(Op(r"a^\dagger", 1), Op("a", -1),
                                mol_list.j_matrix[imol, jmol])]
            # vibration part
            for imol, mol in enumerate(mol_list):
                for iph, ph in enumerate(mol.dmrg_phs):
                    assert np.allclose(np.array(ph.force3rd), [0.0, 0.0])

                    model[(mapping[(imol, iph)],)] = [(Op("p^2", 0), 0.5),
                            (Op("x^2", 0), 0.5*ph.omega[0]**2)]

            # vibration potential part
            for imol, mol in enumerate(mol_list):
                for iph, ph in enumerate(mol.dmrg_phs):
                    if np.allclose(ph.omega[0], ph.omega[1]):
                        model[(f"e_{e_idx(imol)}", f"{mapping[(imol,iph)]}")] = [
                                (Op(r"a^\dagger a", 0), Op("x", 0), -ph.omega[1]**2*ph.dis[1]),
                                ]
                    else:
                        model[(f"e_{e_idx(imol)}", f"{mapping[(imol,iph)]}")] = [
                                (Op(r"a^\dagger a", 0), Op("x^2", 0), 0.5*(ph.omega[1]**2-ph.omega[0]**2)),
                                (Op(r"a^\dagger a", 0), Op("x", 0), -ph.omega[1]**2*ph.dis[1]),
                                ]

            model_translator = ModelTranslator.general_model
        else:
            raise ValueError(f"invalid formula: {formula}")
        
        dipole = {}
        for imol, mol in enumerate(mol_list):
            if mol_list.scheme < 4:
                dipole[(f"e_{e_idx(imol)}", )] = mol.dipole
            elif mol_list.scheme == 4:
                dipole[(f"e_{e_idx(imol)}", "e_0")] = mol.dipole
                dipole[("e_0", f"e_{e_idx(imol)}")] = mol.dipole

        mol_list2 = cls(order, basis, model, model_translator, dipole=dipole)
        mol_list2.map = mapping

        return mol_list2


class MolList:

    def __init__(self, mol_list: List[Mol], j_matrix: Union[Quantity, np.ndarray, None], scheme: int = 2, periodic: bool = False):
        self.periodic = periodic
        self.mol_list: List[Mol] = mol_list

        # construct the electronic coupling matrix
        if j_matrix is None:
            # spin-boson model
            assert len(self.mol_list) == 1
            j_matrix = Quantity(0)

        if isinstance(j_matrix, Quantity):
            self.j_matrix = construct_j_matrix(self.mol_num, j_matrix, periodic)
            self.j_constant = j_matrix
        else:
            if periodic:
                assert j_matrix[0][-1] != 0 and j_matrix[-1][0] != 0
            self.j_matrix = j_matrix
            self.j_constant = None
        self.scheme = scheme

        assert self.j_matrix.shape[0] == self.mol_num

        self.ephtable = EphTable.from_mol_list(mol_list, scheme)
        self.pbond_list: List[int] = []

        if scheme < 4:
            # the order is e0,v00,v01,...,e1,v10,v11,...
            if scheme == 3:
                assert self.check_nearest_neighbour()
            self._e_idx = []
            for mol in mol_list:
                self._e_idx.append(len(self.pbond_list))
                self.pbond_list.append(2)
                for ph in mol.dmrg_phs:
                    self.pbond_list += ph.pbond
        else:
            # the order is v00,v01,..,v10,v11,...,e0/e1/e2,...,v30,v31...
            for imol, mol in enumerate(mol_list):
                if imol == len(mol_list) // 2:
                    self._e_idx = [len(self.pbond_list)] * len(mol_list)
                    self.pbond_list.append(len(mol_list)+1)
                for ph in mol.dmrg_phs:
                    assert ph.is_simple
                    self.pbond_list += ph.pbond

        # reusable mpos for the system
        self.mpos = dict()
        
        # MolList2 type parameter
        # for use in MolList2 routine
        self.order = None
        self.basis = None
        self.model = None
        self.model_translator = None


    @property
    def mol_num(self):
        return len(self.mol_list)
    
    @property
    def nsite(self):
        return len(self.ephtable)

    @property
    def ph_modes_num(self):
        return sum([mol.n_dmrg_phs for mol in self.mol_list])

    @property
    def pure_dmrg(self):
        for mol in self:
            if not mol.pure_dmrg:
                return False
        return True

    @property
    def pure_hartree(self):
        for mol in self:
            if not mol.pure_hartree:
                return False
        return True

    def switch_scheme(self, scheme):
        return self.__class__(self.mol_list.copy(), self.j_matrix.copy(), scheme, self.periodic)

    def copy(self):
        return self.switch_scheme(self.scheme)

    def e_idx(self, idx=0):
        return self._e_idx[idx]

    def ph_idx(self, eidx, phidx):
        if self.scheme < 4:
            start = self.e_idx(eidx)
            assert self.mol_list[eidx].no_qboson
            # skip the electron site
            return start + 1 + phidx
        elif self.scheme == 4:
            res = 0
            for mol in self.mol_list[:eidx]:
                assert mol.no_qboson
                res += mol.n_dmrg_phs
            if self.mol_num // 2 <= eidx:
                res += 1
            return res + phidx
        else:
            assert False

    def get_mpos(self, key, fun):
        if key not in self.mpos:
            mpos = fun(self)
            self.mpos[key] = mpos
        else:
            mpos = self.mpos[key]
        return mpos

    def check_nearest_neighbour(self):
        d = np.diag(np.diag(self.j_matrix, k=1), k=1)
        d = d + d.T
        d[0, -1] = self.j_matrix[-1, 0]
        d[-1, 0] = self.j_matrix[0, -1]
        return np.allclose(d, self.j_matrix)

    def get_sub_mollist(self, span=None):
        assert self.mol_num % 2 == 1
        center_idx = self.mol_num // 2
        if span is None:
            span = self.mol_num // 10
        start_idx = center_idx-span
        end_idx = center_idx+span+1
        sub_list =self.mol_list[start_idx:end_idx]
        sub_matrix = self.j_matrix[start_idx:end_idx, start_idx:end_idx]
        return self.__class__(sub_list, sub_matrix, scheme=self.scheme), start_idx

    def get_pure_dmrg_mollist(self):
        l = []
        for mol in self.mol_list:
            mol = Mol(Quantity(mol.elocalex), mol.dmrg_phs, mol.dipole)
            l.append(mol)
        return self.__class__(l, self.j_matrix, scheme=self.scheme)
    
    def mol_list2_para(self, formula="vibronic"):
        mol_list2 = MolList2.MolList_to_MolList2(self, formula)
        self.order = mol_list2.order
        self.basis = mol_list2.basis
        self.model = mol_list2.model
        if not np.allclose(self.mol_list[0].tunnel, 0):
            #sbm
            self.model_translator = ModelTranslator.sbm
        else:
            if self.scheme == 4:
                self.model_translator = ModelTranslator.Holstein_model_scheme4
            elif self.scheme < 4:
                self.model_translator = ModelTranslator.Holstein_model_scheme123
            else:
                raise ValueError
    
    def rewrite_model(self, model, model_translator):
        mol_list_new = self.__class__(self.mol_list.copy(),
                self.j_matrix.copy(), self.scheme, self.periodic)
        mol_list_new.order = self.order
        mol_list_new.basis = self.basis
        mol_list_new.model = model
        mol_list_new.model_translator = model_translator
        return mol_list_new

    def __getitem__(self, idx):
        return self.mol_list[idx]

    def __len__(self):
        return len(self.mol_list)

    def to_dict(self):
        info_dict = OrderedDict()
        info_dict["mol num"] = len(self)
        info_dict["electron phonon table"] = self.ephtable
        info_dict["mol list"] = [mol.to_dict() for mol in self.mol_list]
        if self.j_constant is None:
            info_dict["J matrix"] = cast_float(self.j_matrix)
        else:
            info_dict["J constant"] = self.j_constant.as_au()
        return info_dict


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
    mol_list = MolList([Mol(Quantity(0), ph_list)] * param["mol num"],
                       j_constant, scheme=scheme
                       )
    return mol_list, temperature
