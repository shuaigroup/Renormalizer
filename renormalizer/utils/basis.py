import numpy as np
from renormalizer.utils import Op
from typing import Dict, List


class BasisSet:
    r"""
    the class for local basis set
    Args:
        nbas (int): # of the basis set
        sigmaqn (List(int)): the qn of each basis
    """
    def __init__(self, nbas: int, sigmaqn: List[int]):
        assert type(nbas) is int
        self.nbas = nbas

        for qn in sigmaqn:
            assert type(qn) is int
        self.sigmaqn = sigmaqn
    
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
            # <m|p|n> = -i sqrt(w/2) <m| b^\dagger - b |n>
            mat = -1j * np.sqrt(self.omega / 2) * (self.op_mat(r"b^\dagger") - self.op_mat("b"))

        elif op_symbol == "p^2":
            mat = -self.omega / 2 * (self.op_mat(r"b^\dagger b^\dagger")
                                     - self.op_mat(r"b^\dagger b")
                                     - self.op_mat(r"b b^\dagger")
                                     + self.op_mat(r"b b")
                                     )

        elif op_symbol == "I":
            mat = np.eye(self.nbas)

        else:
            raise ValueError(f"op_symbol:{op_symbol} is not supported")

        return mat * op_factor


class BasisMultiElectron(BasisSet):
    r"""
    The basis set for multi electronic state on one single site
    Args:
        nstate (int): the # of electronic states
        sigmaqn (List(int)): the sigmaqn of each basis
    """
    
    def __init__(self, nstate, sigmaqn: List[int]):
        
        assert len(sigmaqn) == nstate
        super().__init__(nstate, sigmaqn)

    def op_mat(self, op):
        
        if isinstance(op, Op):
            op_symbol, op_factor = op.symbol, op.factor
        else:
            op_symbol, op_factor = op, 1.0
        
        op_symbol = op_symbol.split(" ")
        
        if len(op_symbol) == 1:
            assert op_symbol[0] == "I"
            mat = np.eye(self.nbas)
            
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
    def __init__(self):
        super().__init__(2, [0,0])

    def op_mat(self, op):
        
        if isinstance(op, Op):
            op_symbol, op_factor = op.symbol, op.factor
        else:
            op_symbol, op_factor = op, 1.0
        
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
        else:
            raise ValueError(f"op_symbol:{op_symbol} is not supported")

        return mat * op_factor
