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


class Basis_SHO(BasisSet):
    """
    simple harmonic oscillator basis set
    Args:
        omega (float): the frequency of the oscillator.
        x0 (float): the origin of the harmonic oscillator. Default = 0.
    """
    
    def __init__(self, omega, nbas, x0=0.):
        self.omega = omega
        self.x0 = x0  # origin = x0
        super().__init__(nbas, [0,] * nbas)
        
    def __repr__(self):
        return f"(x0: {self.x0}, omega: {self.omega}, nbas: {self.nbas})"
    
    def op_mat(self, op):
        if isinstance(op, Op):
            op_symbol, op_factor = op.symbol, op.factor
        else:
            op_symbol, op_factor = op, 1.0

        if op_symbol in ["b", r"b^\dagger", r"b^\dagger b", r"b^\dagger + b"]:
            assert np.allclose(self.x0, 0)

        if op_symbol == "x":
            # define x-x0 = y or x = y+x0

            mat = np.eye(self.nbas) * self.x0
            # <m|y|n> = sqrt(1/2w) <m|b^dagger b |n> = sqrt(n+1) delta(m, n+1) +
            # sqrt(n) delta(m, n-1)
            for iket in range(self.nbas-1):
                mat[iket+1, iket] += np.sqrt(iket+1) * np.sqrt(0.5/self.omega)
            for iket in range(1, self.nbas):
                mat[iket-1, iket] += np.sqrt(iket) * np.sqrt(0.5/self.omega)

        elif op_symbol == "x^2":
            # x0^2
            mat = np.eye(self.nbas) * self.x0**2
            
            # 2 x0 * y 
            for iket in range(self.nbas-1):
                mat[iket+1, iket] += np.sqrt(iket+1) * np.sqrt(0.5/self.omega) * 2. * self.x0
            for iket in range(1, self.nbas):
                mat[iket-1, iket] += np.sqrt(iket) * np.sqrt(0.5/self.omega) * 2. * self.x0
            
            #  y^2: b^dagger b + b b^\dagger = (n+1+n) delta(m,n)
            for iket in range(self.nbas):
                mat[iket, iket] += (2*iket+1) * 0.5/self.omega
            
            #  y^2: bb = sqrt(n*(n-1)) delta(m,n-2)
            for iket in range(2, self.nbas):
                mat[iket-2, iket] += np.sqrt(iket*(iket-1)) * 0.5/self.omega

            #  y^2: b^\dagger b^\dagger = sqrt((n+2)*(n+1)) delta(m,n+2)
            for iket in range(0, self.nbas-2):
                mat[iket+2, iket] += np.sqrt((iket+2)*(iket+1)) * 0.5/self.omega
        
        elif op_symbol == "p":
            #<m|p|n> = -i sqrt(w/2) sqrt(n) delta(m,n-1) - sqrt(n+1) delta(m,n+1)
            mat = np.zeros((self.nbas, self.nbas), dtype=np.complex128)
            
            for iket in range(1, self.nbas):
                mat[iket-1, iket] += -1.0j * np.sqrt(0.5*self.omega) * np.sqrt(iket) 
            for iket in range(self.nbas-1):
                mat[iket+1, iket] -= -1.0j * np.sqrt(0.5*self.omega) * np.sqrt(iket+1)

        elif op_symbol == "p^2":
            
            mat = np.zeros((self.nbas, self.nbas))
            
            #  -(b^dagger b + b b^\dagger) = (n+1+n) delta(m,n)
            for iket in range(self.nbas):
                mat[iket, iket] -= (-0.5*self.omega) * (2*iket+1)  
            
            #  bb = sqrt(n*(n-1)) delta(m,n-2)
            for iket in range(2, self.nbas):
                mat[iket-2, iket] += (-0.5*self.omega) * np.sqrt(iket*(iket-1))  

            #  b^\dagger b^\dagger = sqrt((n+2)*(n+1)) delta(m,n+2)
            for iket in range(0, self.nbas-2):
                mat[iket+2, iket] += (-0.5*self.omega) * np.sqrt((iket+2)*(iket+1)) 

        elif op_symbol == "I":
            mat = np.eye(self.nbas)
        
        # second quantization formula
        elif op_symbol == "b":
            mat = np.zeros((self.nbas, self.nbas))
            # b = sqrt(n) delta(n-1,n)
            for iket in range(1, self.nbas):
                mat[iket-1, iket] = np.sqrt(iket) 

        elif op_symbol == r"b^\dagger":
            mat = np.zeros((self.nbas, self.nbas))
            # b^\dagger = sqrt(n+1) delta(n+1,n)
            for iket in range(0, self.nbas-1):
                mat[iket+1, iket] = np.sqrt(iket+1) 
        
        elif op_symbol == r"b^\dagger + b":
            mat = np.zeros((self.nbas, self.nbas))
            # b = sqrt(n) delta(n-1,n)
            for iket in range(1, self.nbas):
                mat[iket-1, iket] = np.sqrt(iket) 
            # b^\dagger = sqrt(n+1) delta(n+1,n)
            for iket in range(0, self.nbas-1):
                mat[iket+1, iket] = np.sqrt(iket+1) 
        
        elif op_symbol == r"b^\dagger b":
            mat = np.zeros((self.nbas, self.nbas))
            # b^dagger b = n delta(n,n)
            for iket in range(self.nbas):
                mat[iket, iket] = float(iket)

        else:
            raise ValueError(f"op_symbol:{op_symbol} is not supported")

        return mat * op_factor


class Basis_Multi_Electron(BasisSet):
    r"""
    The basis set for multi electronic state on one single site
    Args:
        nstate (int): the # of electronic states
        sigmaqn (List(int)): the sigmaqn of each basis
    """
    
    def __init__(self, nstate, sigmaqn:List[int]):
        
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


class Basis_Simple_Electron(BasisSet):
    r"""
    The basis set for simple electron DoF, two state with 0: unoccupied, 1: occupied

    """
    def __init__(self):
        super().__init__(2, [0,1])

    def op_mat(self, op):
        
        if isinstance(op, Op):
            op_symbol, op_factor = op.symbol, op.factor
        else:
            op_symbol, op_factor = op, 1.0
        
        mat = np.zeros((2,2))
        
        if op_symbol == r"a^\dagger":
            mat[1,0] = 1.
        
        elif  op_symbol == "a":
            mat[0,1] = 1.
        
        elif op_symbol == r"a^\dagger a":
            mat[1,1] = 1.
        
        elif op_symbol == "I":
            mat = np.eye(2)

        else:
            raise ValueError(f"op_symbol:{op_symbol} is not supported")
        
        return mat * op_factor


class Basis_Half_Spin(BasisSet):
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
