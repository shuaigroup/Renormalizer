import numpy as np
from renormalizer.utils import Op
from typing import Dict, List


class BasisSet:
    r"""
    the parent class for local basis set

    Args:
        nbas (int): number of dimension of the basis set
        sigmaqn (List(int)): the qn of each basis
        multi_dof (bool) : if multiple dof is contained in this basis
    """
    def __init__(self, nbas: int, sigmaqn: List[int], multi_dof=False):
        assert type(nbas) is int
        self.nbas = nbas

        for qn in sigmaqn:
            assert type(qn) is int
        self.sigmaqn = sigmaqn
        self.multi_dof = multi_dof
    
    def __repr__(self):
        return f"(nbas: {self.nbas}, qn: {self.sigmaqn})"
    
    def op_mat(self, op):
        raise NotImplementedError


class BasisSHO(BasisSet):
    """
    simple harmonic oscillator basis set

    Args:
        omega (float): the frequency of the oscillator.
        x0 (float): the origin of the harmonic oscillator. Default = 0.
    """
    
    def __init__(self, omega, nbas, x0=0.):
        self.omega = omega
        self.x0 = x0  # origin = x0
        super().__init__(nbas, [0] * nbas)
        
    def __repr__(self):
        return f"(x0: {self.x0}, omega: {self.omega}, nbas: {self.nbas})"
    
    def op_mat(self, op):
        if isinstance(op, Op):
            op_symbol, op_factor = op.symbol, op.factor
        else:
            op_symbol, op_factor = op, 1.0

        if op_symbol in ["b", "b b", r"b^\dagger", r"b^\dagger b^\dagger", r"b^\dagger b", r"b b^\dagger", r"b^\dagger + b"]:
            assert np.allclose(self.x0, 0)

        # so many if-else might be a potential performance problem in the future
        # change to lazy-evaluation dict should be better

        # second quantization formula
        if op_symbol == "b":
            mat = np.diag(np.sqrt(np.arange(1, self.nbas)), k=1)

        elif op_symbol == "b b":
            # b b = sqrt(n*(n-1)) delta(m,n-2)
            mat = np.diag(np.sqrt(np.arange(1, self.nbas - 1) * np.arange(2, self.nbas)), k=2)

        elif op_symbol == r"b^\dagger":
            mat = np.diag(np.sqrt(np.arange(1, self.nbas)), k=-1)

        elif op_symbol == r"b^\dagger b^\dagger":
            # b^\dagger b^\dagger = sqrt((n+2)*(n+1)) delta(m,n+2)
            mat = np.diag(np.sqrt(np.arange(1, self.nbas - 1) * np.arange(2, self.nbas)), k=-2)

        elif op_symbol == r"b^\dagger + b":
            mat = self.op_mat(r"b^\dagger") + self.op_mat("b")

        elif op_symbol == r"b^\dagger b":
            # b^dagger b = n delta(n,n)
            mat = np.diag(np.arange(self.nbas))

        elif op_symbol == r"b b^\dagger":
            mat = np.diag(np.arange(self.nbas) + 1)

        elif op_symbol == "x":
            # define x-x0 = y or x = y+x0, return x
            # <m|y|n> = sqrt(1/2w) <m| b^\dagger + b |n>
            mat = np.sqrt(0.5/self.omega) * self.op_mat(r"b^\dagger + b") + np.eye(self.nbas) * self.x0

        elif op_symbol == "x^2":
            # can't do things like the commented code below due to numeric error around highest quantum number
            # x_mat = self.op_mat("x")
            # mat = x_mat @ x_mat

            # x^2 = x0^2 + 2 x0 * y + y^2
            # x0^2
            mat = np.eye(self.nbas) * self.x0**2

            # 2 x0 * y
            mat += 2 * self.x0 * np.sqrt(0.5/self.omega) * self.op_mat(r"b^\dagger + b")

            #  y^2: 1/2w * (b^\dagger b^\dagger + b^dagger b + b b^\dagger + bb)
            mat += 0.5/self.omega * (self.op_mat(r"b^\dagger b^\dagger")
                                     + self.op_mat(r"b^\dagger b")
                                     + self.op_mat(r"b b^\dagger")
                                     + self.op_mat(r"b b")
                                     )

        elif op_symbol == "p":
            # <m|p|n> = -i sqrt(w/2) <m| b - b^\dagger |n>
            mat = 1j * np.sqrt(self.omega / 2) * (self.op_mat(r"b^\dagger") - self.op_mat("b"))

        elif op_symbol == "p^2":
            mat = -self.omega / 2 * (self.op_mat(r"b^\dagger b^\dagger")
                                     - self.op_mat(r"b^\dagger b")
                                     - self.op_mat(r"b b^\dagger")
                                     + self.op_mat(r"b b")
                                     )

        elif op_symbol == "I":
            mat = np.eye(self.nbas)
        
        elif op_symbol == "n":
            # since b^\dagger b is not allowed to shift the origin, 
            # n is designed for occupation number of the SHO basis
            mat = np.diag(np.arange(self.nbas))
        else:
            raise ValueError(f"op_symbol:{op_symbol} is not supported")

        return mat * op_factor


class BasisMultiElectron(BasisSet):
    r"""
    The basis set for multi electronic state on one single site,
    the basis order is ["e_0", "e_1", "e_2",...]

    Args:
        nstate (int): the # of electronic states
        sigmaqn (List(int)): the sigmaqn of each basis
    """
    
    def __init__(self, nstate, sigmaqn: List[int]):
        
        assert len(sigmaqn) == nstate
        super().__init__(nstate, sigmaqn, True)

    def op_mat(self, op):
        
        if isinstance(op, Op):
            op_symbol, op_factor = op.symbol, op.factor
        else:
            op_symbol, op_factor = op, 1.0
        
        op_symbol = op_symbol.split(" ")
        
        if len(op_symbol) == 1:
            if op_symbol[0] == "I":
                mat = np.eye(self.nbas)
            elif op_symbol[0] == "a" or op_symbol[0] == r"a^\dagger":
                raise ValueError(f"op_symbol:{op_symbol} is not supported. Try use BasisMultiElectronVac.")
            
        elif len(op_symbol) == 2:
            op_symbol1, op_symbol2 = op_symbol
            op_symbol1_term, op_symbol1_idx = op_symbol1.split("_")
            op_symbol2_term, op_symbol2_idx = op_symbol2.split("_")

            mat = np.zeros((self.nbas, self.nbas))
            
            if op_symbol1_term == r"a^\dagger" and op_symbol2_term == "a":
                mat[int(op_symbol1_idx), int(op_symbol2_idx)] = 1.
            elif op_symbol1_term == r"a" and op_symbol2_term == r"a^\dagger":
                mat[int(op_symbol2_idx), int(op_symbol1_idx)] = 1.
            else:
                assert False
        else:
            raise ValueError(f"op_symbol:{op_symbol} is not supported")

        return mat * op_factor


class BasisMultiElectronVac(BasisSet):
    r"""
    Another basis set for multi electronic state on one single site.
    Vacuum state is included.
    The basis order is [vacuum, "e_0", "e_1", "e_2",...].
    sigma qn is [0, 1, 1, 1, ...]

    Args:
        nstate (int): the number of electronic states without counting the vacuum state.
        dof_idx (list): the indices of the electronic dofs used to define the whole model
        that are represented by this basis. The default value is ``list(range(nstate))``.
        The arg is necessary when you have multiple basis of this class in the model.
    """

    def __init__(self, nstate, dof_idx=None):

        sigmaqn = [0] + [1] * nstate
        if dof_idx is None:
            dof_idx = list(range(nstate))
        # map external dof index into internal dof index
        # the index 0 is reserved for vacuum
        self.dof_idx_map = {k: v+1 for v, k in enumerate(dof_idx)}
        super().__init__(nstate+1, sigmaqn, True)

    def _split_sym_idx(self, op_symbol):
        op_symbol_term, op_symbol_idx = op_symbol.split("_")
        op_symbol_idx = self.dof_idx_map[int(op_symbol_idx)]
        return op_symbol_term, op_symbol_idx

    def op_mat(self, op):

        if isinstance(op, Op):
            op_symbol, op_factor = op.symbol, op.factor
        else:
            op_symbol, op_factor = op, 1.0

        op_symbol = op_symbol.split(" ")

        mat = None
        if len(op_symbol) == 1:
            op_symbol = op_symbol[0]
            if op_symbol == "I":
                mat = np.eye(self.nbas)
            else:
                mat = np.zeros((self.nbas, self.nbas))
                op_symbol_term, op_symbol_idx = self._split_sym_idx(op_symbol)
                if op_symbol_term == r"a^\dagger":
                    mat[op_symbol_idx, 0] = 1.
                elif op_symbol_term == r"a":
                    mat[0, op_symbol_idx] = 1.

        elif len(op_symbol) == 2:
            op_symbol1, op_symbol2 = op_symbol
            op_symbol1_term, op_symbol1_idx = self._split_sym_idx(op_symbol1)
            op_symbol2_term, op_symbol2_idx = self._split_sym_idx(op_symbol2)

            mat = np.zeros((self.nbas, self.nbas))

            if op_symbol1_term == r"a^\dagger" and op_symbol2_term == "a":
                mat[op_symbol1_idx, op_symbol2_idx] = 1.
            elif op_symbol1_term == r"a" and op_symbol2_term == r"a^\dagger":
                mat[op_symbol2_idx, op_symbol1_idx] = 1.

        if mat is None:
            raise ValueError(f"op_symbol:{op_symbol} is not supported")

        return mat * op_factor


class BasisSimpleElectron(BasisSet):
    r"""
    The basis set for simple electron DoF, two state with 0: unoccupied, 1: occupied

    """
    def __init__(self):
        super().__init__(2, [0, 1])

    def op_mat(self, op):
        
        if isinstance(op, Op):
            op_symbol, op_factor = op.symbol, op.factor
        else:
            op_symbol, op_factor = op, 1.0
        
        mat = np.zeros((2, 2))
        
        if op_symbol == r"a^\dagger":
            mat[1, 0] = 1.
        elif op_symbol == "a":
            mat[0, 1] = 1.
        elif op_symbol == r"a^\dagger a":
            mat[1, 1] = 1.
        elif op_symbol == "I":
            mat = np.eye(2)
        else:
            raise ValueError(f"op_symbol:{op_symbol} is not supported")
        
        return mat * op_factor


class BasisHalfSpin(BasisSet):
    r"""
    The basis the for 1/2 spin DoF
    """
    def __init__(self, sigmaqn:List[int]=None):
        if sigmaqn is None:
            sigmaqn = [0, 0]
        super().__init__(2, sigmaqn)

    def op_mat(self, op):
        
        if isinstance(op, Op):
            op_symbol, op_factor = op.symbol, op.factor
        else:
            op_symbol, op_factor = op, 1.0
        
        op_symbol = op_symbol.split(" ")
        
        if len(op_symbol) == 1:       
            op_symbol = op_symbol[0]
            
            if op_symbol == "I":
                mat = np.eye(2)
            elif op_symbol == "sigma_x":
                mat = np.diag([1.], k=1)
                mat = mat + mat.T.conj()
            elif op_symbol == "sigma_y":
                mat = np.diag([-1.0j], k=1)
                mat = mat + mat.T.conj()
            elif op_symbol == "sigma_z":
                mat = np.diag([1.,-1.], k=0)
            elif op_symbol == "sigma_-":
                mat = np.diag([1.], k=-1)
            elif op_symbol == "sigma_+":
                mat = np.diag([1.,], k=1)
            else:
                raise ValueError(f"op_symbol:{op_symbol} is not supported")
        else:
            mat = np.eye(2)
            for o in op_symbol:
                mat = mat @ self.op_mat(o)

        return mat * op_factor
