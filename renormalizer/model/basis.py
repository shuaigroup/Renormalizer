import numpy as np
from renormalizer.model import Op
from typing import Union, List
import scipy.linalg
import scipy.special
import itertools
import logging


logger = logging.getLogger(__name__)


class BasisSet:
    r"""
    the parent class for local basis set

    Args:
        dof_name: The name(s) of the DoF(s) contained in the basis set.
            For basis containing only one DoF, the type could be anything that can be hashed.
            For basis containing multiple DoFs, the type should be a ``list`` or ``tuple``
            of anything that can be hashed.
        nbas (int): number of dimension of the basis set
        sigmaqn (List(int)): the qn of each basis

    """

    #: If the basis set represent electronic DoF.
    is_electron = False
    #: If the basis set represent vibrational DoF.
    is_phonon = False
    #: If the basis set represent spin DoF.
    is_spin = False
    #: If the basis set contains multiple DoFs.
    multi_dof = False

    def __init__(self, dof, nbas: int, sigmaqn: List[int]):
        self.dof = dof

        assert type(nbas) is int
        self.nbas = nbas

        for qn in sigmaqn:
            assert type(qn) is int
        self.sigmaqn = sigmaqn

    def __repr__(self):
        return f"(nbas: {self.nbas}, qn: {self.sigmaqn})"
    
    def op_mat(self, op: Op):
        """
        Matrix representation under the basis set of the input operator.
        The factor is included.

        Parameters
        ----------
        op : Op
            The operator. For basis set with only one DoF, :class:`str`` is also acceptable.

        Returns
        -------
        mat : :class:`np.ndarray`
            Matrix representation of ``op``.
        """
        raise NotImplementedError

    @property
    def dofs(self):
        """
        Names of the DoFs contained in the basis.
        Returns a tuple even if the basis contains only one DoF.

        Returns
        -------
        dof names : tuple
            A tuple of DoF names.
        """
        if self.multi_dof:
            return tuple(self.dof)
        else:
            return (self.dof,)


    def copy(self, new_dof):
        """
        Return a copy of the basis set with new DoF name specified in the argument.

        Parameters
        ----------
        new_dof:
            New DoF name.

        Returns
        -------
        new_basis : Basis
            A copy of the basis with new DoF name.
        """
        raise NotImplementedError


class BasisSHO(BasisSet):
    """
    simple harmonic oscillator basis set

    Args:
        dof: The name of the DoF contained in the basis set. The type could be anything that can be hashed.
        omega (float): the frequency of the oscillator.
        nbas (int): number of dimension of the basis set (highest occupation number of the harmonic oscillator)
        x0 (float): the origin of the harmonic oscillator. Default = 0.
        dvr (bool): whether to use discrete variable representation. Default = False.
        general_xp_power (bool): whether calculate :math:`x` and :math:`x^2` (or :math:`p` and :math:`p^2`)
            through general expression for :math:`x`power or :math:`p` power. This is not efficient because
            :math:`x` and :math:`x^2` (or :math:`p` and :math:`p^2`) have been hard-coded already.
            The option is only used for testing.
    """

    is_phonon = True

    def __init__(self, dof, omega, nbas, x0=0., dvr=False, general_xp_power=False):
        self.omega = omega
        self.x0 = x0  # origin = x0
        super().__init__(dof, nbas, [0] * nbas)

        self.general_xp_power = general_xp_power

        # whether under recurssion
        self._recurssion_flag = 0

        self.dvr = False
        self.dvr_x = None  # the expectation value of x on SHO_dvr
        self.dvr_v = None  # the rotation matrix between SHO to SHO_dvr
        if dvr:
            self.dvr_x, self.dvr_v = scipy.linalg.eigh(self.op_mat("x"))
            self.dvr = True

    def __repr__(self):
        return f"(x0: {self.x0}, omega: {self.omega}, nbas: {self.nbas})"
    
    def op_mat(self, op: Union[Op, str]):
        if not isinstance(op, Op):
            op = Op(op, None)
        op_symbol, op_factor = op.symbol, op.factor

        if op_symbol in ["b", "b b", r"b^\dagger", r"b^\dagger b^\dagger", r"b^\dagger b", r"b b^\dagger", r"b^\dagger + b"]:
            if self._recurssion_flag == 0 and not np.allclose(self.x0, 0):
                logger.warning("the second quantization doesn't support nonzero x0")

        self._recurssion_flag += 1

        # so many if-else might be a potential performance problem in the future
        # changing to lazy-evaluation dict should be better
        
        # second quantization formula
        if op_symbol == "b":
            mat = np.diag(np.sqrt(np.arange(1, self.nbas)), k=1)

        elif op_symbol == "b b":
            # b b = sqrt(n*(n-1)) delta(m,n-2)
            if self.nbas == 1:
                mat = np.zeros((1,1))
            else:
                mat = np.diag(np.sqrt(np.arange(1, self.nbas - 1) * np.arange(2, self.nbas)), k=2)

        elif op_symbol == r"b^\dagger":
            mat = np.diag(np.sqrt(np.arange(1, self.nbas)), k=-1)

        elif op_symbol == r"b^\dagger b^\dagger":
            # b^\dagger b^\dagger = sqrt((n+2)*(n+1)) delta(m,n+2)
            if self.nbas == 1:
                mat = np.zeros((1,1))
            else:
                mat = np.diag(np.sqrt(np.arange(1, self.nbas - 1) * np.arange(2, self.nbas)), k=-2)

        elif op_symbol == r"b^\dagger + b":
            mat = self.op_mat(r"b^\dagger") + self.op_mat("b")

        elif op_symbol == r"b^\dagger b":
            # b^dagger b = n delta(n,n)
            mat = np.diag(np.arange(self.nbas))

        elif op_symbol == r"b b^\dagger":
            mat = np.diag(np.arange(self.nbas) + 1)

        elif op_symbol == "x" and (not self.general_xp_power):
            if not self.dvr:
                # define x-x0 = y or x = y+x0, return x
                # <m|y|n> = sqrt(1/2w) <m| b^\dagger + b |n>
                mat = np.sqrt(0.5/self.omega) * self.op_mat(r"b^\dagger + b") + np.eye(self.nbas) * self.x0
            else:
                mat = np.diag(self.dvr_x)

        elif op_symbol == "x^2" and (not self.general_xp_power):
            
            if not self.dvr:
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
            else:
                mat = np.diag(self.dvr_x**2)
        elif set(op_symbol.split(" ")) == set("x"):
            moment = len(op_symbol.split(" "))
            mat = self.op_mat(f"x^{moment}")
        
        elif op_symbol.split("^")[0] == "x":
            # moments of x
            if len(op_symbol.split("^")) == 1:
                moment = 1
            else:
                moment = float(op_symbol.split("^")[1])
            
            if not self.dvr:
                # Analytical expression for integer moment
                assert np.allclose(moment, round(moment)) 
                moment = round(moment)
                mat = np.zeros((self.nbas, self.nbas))
                for imoment in range(moment+1):
                    factor = scipy.special.comb(moment, imoment) * np.sqrt(1/self.omega) ** imoment
                    for i,j in itertools.product(range(self.nbas), repeat=2):
                        mat[i,j] += factor * x_power_k(imoment, i, j) * self.x0**(moment-imoment)
                
            else:
                mat = np.diag(self.dvr_x ** moment)

        elif op_symbol == "p" and (not self.general_xp_power):
            # <m|p|n> = -i sqrt(w/2) <m| b - b^\dagger |n>
            mat = 1j * np.sqrt(self.omega / 2) * (self.op_mat(r"b^\dagger") - self.op_mat("b"))
            if self.dvr:
                mat = self.dvr_v.T @ mat @ self.dvr_v

        elif op_symbol == "p^2" and (not self.general_xp_power):
            mat = -self.omega / 2 * (self.op_mat(r"b^\dagger b^\dagger")
                                     - self.op_mat(r"b^\dagger b")
                                     - self.op_mat(r"b b^\dagger")
                                     + self.op_mat(r"b b")
                                     )
            if self.dvr:
                mat = self.dvr_v.T @ mat @ self.dvr_v

        elif set(op_symbol.split(" ")) == set("p"):
            moment = len(op_symbol.split(" "))
            mat = self.op_mat(f"p^{moment}")

        elif op_symbol.split("^")[0] == "p":
            # moments of p
            if len(op_symbol.split("^")) == 1:
                moment = 1
            else:
                moment = float(op_symbol.split("^")[1])
            
            # the moment for p should be integer
            assert np.allclose(moment, round(moment)) 
            moment = round(moment)
            if moment % 2 == 0:
                dtype = np.float64
            else:
                dtype = np.complex128
            mat = np.zeros((self.nbas, self.nbas), dtype=dtype)

            for i,j in itertools.product(range(self.nbas), repeat=2):
                res = p_power_k(moment, i, j) * np.sqrt(self.omega) ** moment
                if moment % 2 == 0:
                    mat[i,j] = np.real(res)
                else:
                    mat[i,j] = res
                
            if self.dvr:
                mat = self.dvr_v.T @ mat @ self.dvr_v
        
        elif op_symbol == "x p":
            mat = -1.0j/2 *(self.op_mat(r"b b")
                    - self.op_mat(r"b^\dagger b^\dagger")
                    + self.op_mat(r"b b^\dagger")
                    - self.op_mat(r"b^\dagger b"))
        
        elif op_symbol == "x partialx":
            # x partialx is real, while x p is imaginary
            mat = (self.op_mat("x p") / -1.0j).real
        
        elif op_symbol == "p x":
            mat = -1.0j/2 *(self.op_mat(r"b b")
                    - self.op_mat(r"b^\dagger b^\dagger")
                    - self.op_mat(r"b b^\dagger")
                    + self.op_mat(r"b^\dagger b"))
        
        elif op_symbol == "partialx x":
            mat = (self.op_mat("p x") / -1.0j).real

        elif op_symbol == "partialx":
            mat = (self.op_mat("p") / -1.0j).real
        
        elif op_symbol in ["partialx^2", "partialx partialx"]:
            mat = self.op_mat("p^2") * -1
        
        elif op_symbol == "I":
            mat = np.eye(self.nbas)
        
        elif op_symbol == "n":
            # since b^\dagger b is not allowed to shift the origin, 
            # n is designed for occupation number of the SHO basis
            mat = np.diag(np.arange(self.nbas))
        else:
            raise ValueError(f"op_symbol:{op_symbol} is not supported. ")

        self._recurssion_flag -= 1
        return mat * op_factor

    def copy(self, new_dof):
        return self.__class__(new_dof, omega=self.omega,
                              nbas=self.nbas, x0=self.x0,
                              dvr=self.dvr, general_xp_power=self.general_xp_power)


class BasisSineDVR(BasisSet):
    r"""
    Sine DVR basis (particle-in-a-box) for vibrational, angular, and
    dissociative modes.
    See Phys. Rep. 324, 1–105 (2000).

        .. math::
            \psi_j(x) = \sqrt{\frac{2}{L}} \sin(j\pi(x-x_0)/L) \, \textrm{for} \, x_0 \le x \le
            x_{N+1}, L = x_{N+1} - x_0

    the grid points are at

        .. math::
            x_\alpha = x_0 + \alpha \frac{L}{N+1}
    
    Parameters
    ----------
    dof: str, int 
        The name of the DoF contained in the basis set. The type could be anything that can be hashed.
    nbas: int
        Number of grid points.
    xi: float
        The leftmost grid point of the coordinate.
    xf: float
        The rightmost grid point of the coordinate.
    endpoint: bool, optional
        If ``endpoint=False``, :math:`x_0=x_i, x_{N+1}=x_f`; otherwise
        :math:`x_1=x_i, x_{N}=x_f`.
            
    """
    is_phonon = True

    def __init__(self, dof, nbas, xi, xf, endpoint=False):
        if endpoint:
            interval = (xf-xi) / (nbas-1)
            xi -= interval
            xf += interval

        self.xi = xi
        self.xf = xf

        self.L = xf-xi
        super().__init__(dof, nbas, [0] * nbas)
        
        tmp = np.arange(1,nbas+1)
        self.dvr_x = xi + tmp * self.L / (nbas+1)
        self.dvr_v = np.sqrt(2/(nbas+1)) * \
            np.sin(np.tensordot(tmp, tmp, axes=0)*np.pi/(nbas+1))

    def __repr__(self):
        return f"(xi: {self.xi}, xf: {self.xf}, nbas: {self.nbas})"
    
    def op_mat(self, op: Union[Op, str]):
        if not isinstance(op, Op):
            op = Op(op, None)
        op_symbol, op_factor = op.symbol, op.factor
        
        if op_symbol == "I":
            mat = np.eye(self.nbas)
        
        elif op_symbol.split("^")[0] == "x":
            if len(op_symbol.split("^")) == 1:
                moment = 1
            else:
                moment = float(op_symbol.split("^")[1])
            mat = np.diag(self.dvr_x ** moment)
        
        elif set(op_symbol.split(" ")) == set("x"):
            moment = len(op_symbol.split(" "))
            mat = self.op_mat(f"x^{moment}")
        
        elif op_symbol == "partialx":
            mat = np.zeros((self.nbas, self.nbas))
            for j in range(self.nbas):
                for k in range(j):
                    if (j-k) % 2 != 0:
                        mat[j,k] = 4 / self.L * (j+1) * (k+1) / ((j+1)**2 - (k+1)**2)
            mat -= mat.T 
            mat = self.dvr_v.T @ mat @ self.dvr_v

        elif op_symbol in ["partialx^2", "partialx partialx"]:
            mat = -np.diag(np.arange(1, self.nbas+1)*np.pi/self.L)**2
            mat = self.dvr_v.T @ mat @ self.dvr_v

        elif op_symbol == "p":
            mat = self.op_mat("partialx") * -1.0j

        elif op_symbol == "p^2":
            mat = self.op_mat("partialx^2") * -1
        
        else:
            raise ValueError(f"op_symbol:{op_symbol} is not supported. ")

        return mat * op_factor

    def copy(self, new_dof):
        return self.__class__(new_dof, self.nbas, xi=self.xi, xf=self.xf)


class BasisMultiElectron(BasisSet):
    r"""
    The basis set for multi electronic state on one single site,
    The basis order is [dof_names[0], dof_names[1], dof_names[2], ...].

    Parameters
    ----------
    dof : a :class:`list` or :class:`tuple` of hashable objects.
        The names of the DoFs contained in the basis set.
    sigmaqn : :class:`list` of :class:`int`
        The sigmaqn of each basis
    """

    is_electron = True
    multi_dof = True
    
    def __init__(self, dof, sigmaqn: List[int]):
        
        assert len(sigmaqn) == len(sigmaqn)
        self.dof_name_map = {name: i for i, name in enumerate(dof)}
        super().__init__(dof, len(dof), sigmaqn)

    def op_mat(self, op: Op):

        op_symbol, op_factor = op.split_symbol, op.factor
        
        if len(op_symbol) == 1:
            if op_symbol[0] == "I":
                mat = np.eye(self.nbas)
            elif op_symbol[0] == "a" or op_symbol[0] == r"a^\dagger":
                raise ValueError(f"op_symbol:{op_symbol} is not supported. Try use BasisMultiElectronVac.")
            else:
                raise ValueError(f"op_symbol:{op_symbol} is not supported")
            
        elif len(op_symbol) == 2:
            op_symbol1, op_symbol2 = op_symbol
            op_symbol1_idx = self.dof_name_map[op.dofs[0]]
            op_symbol2_idx = self.dof_name_map[op.dofs[1]]

            mat = np.zeros((self.nbas, self.nbas))
            
            if op_symbol1 == r"a^\dagger" and op_symbol2 == "a":
                mat[int(op_symbol1_idx), int(op_symbol2_idx)] = 1.
            elif op_symbol1 == r"a" and op_symbol2 == r"a^\dagger":
                mat[int(op_symbol2_idx), int(op_symbol1_idx)] = 1.
            else:
                raise ValueError(f"op_symbol:{op_symbol} is not supported")
        else:
            raise ValueError(f"op_symbol:{op_symbol} is not supported")

        return mat * op_factor

    def copy(self, new_dof):
        return self.__class__(new_dof, self.sigmaqn)


class BasisMultiElectronVac(BasisSet):
    r"""
    Another basis set for multi electronic state on one single site.
    Vacuum state is included.
    The basis order is [vacuum, dof_names[0], dof_names[1],...].
    sigma qn is [0, 1, 1, 1, ...]

    Parameters
    ----------
    dof : a :class:`list` or :class:`tuple` of hashable objects.
        The names of the DoFs contained in the basis set.
    """

    is_electron = True
    multi_dof = True

    def __init__(self, dof):

        sigmaqn = [0] + [1] * len(dof)
        # map external dof index into internal dof index
        # the index 0 is reserved for vacuum
        self.dof_name_map = {k: v + 1 for v, k in enumerate(dof)}
        super().__init__(dof, len(dof) + 1, sigmaqn)

    def op_mat(self, op: Op):

        op_symbol, op_factor = op.split_symbol, op.factor

        if len(op_symbol) == 1:
            op_symbol = op_symbol[0]
            if op_symbol == "I":
                mat = np.eye(self.nbas)
            else:
                mat = np.zeros((self.nbas, self.nbas))
                op_symbol_idx = self.dof_name_map[op.dofs[0]]
                if op_symbol == r"a^\dagger":
                    mat[op_symbol_idx, 0] = 1.
                elif op_symbol == r"a":
                    mat[0, op_symbol_idx] = 1.
                else:
                    raise ValueError(f"op_symbol:{op_symbol} is not supported")

        elif len(op_symbol) == 2:
            op_symbol1, op_symbol2 = op_symbol
            op_symbol1_idx = self.dof_name_map[op.dofs[0]]
            op_symbol2_idx = self.dof_name_map[op.dofs[1]]

            mat = np.zeros((self.nbas, self.nbas))

            if op_symbol1 == r"a^\dagger" and op_symbol2 == "a":
                mat[op_symbol1_idx, op_symbol2_idx] = 1.
            elif op_symbol1 == r"a" and op_symbol2 == r"a^\dagger":
                mat[op_symbol2_idx, op_symbol1_idx] = 1.
            else:
                raise ValueError(f"op_symbol:{op_symbol} is not supported")
        else:
            raise ValueError(f"op_symbol:{op_symbol} is not supported")

        return mat * op_factor

    def copy(self, new_dof):
        return self.__class__(new_dof)

class BasisSimpleElectron(BasisSet):

    r"""
    The basis set for simple electron DoF, two state with 0: unoccupied, 1: occupied

    Parameters
    ----------
    dof : any hashable object
        The name of the DoF contained in the basis set.

    """
    is_electron = True

    def __init__(self, dof):
        super().__init__(dof, 2, [0, 1])

    def op_mat(self, op):
        if not isinstance(op, Op):
            op = Op(op, None)
        op_symbol, op_factor = op.symbol, op.factor
        
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

    def copy(self, new_dof):
        return self.__class__(new_dof)


class BasisHalfSpin(BasisSet):
    r"""
    The basis the for 1/2 spin DoF

    Parameters
    ----------
    dof : any hashable object
        The name of the DoF contained in the basis set.
    """

    is_spin = True

    def __init__(self, dof, sigmaqn:List[int]=None):
        if sigmaqn is None:
            sigmaqn = [0, 0]
        super().__init__(dof, 2, sigmaqn)

    def op_mat(self, op: Union[Op, str]):
        if not isinstance(op, Op):
            op = Op(op, None)
        op_symbol, op_factor = op.split_symbol, op.factor
        
        if len(op_symbol) == 1:       
            op_symbol = op_symbol[0]
            # Todo: the string itself may sometimes consume a lot of memory when
            # the number of terms in O is very huge
            # replace sigma_x with s_x in the future
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

    def copy(self, new_dof):
        return self.__class__(new_dof, self.sigmaqn)


def x_power_k(k, m, n):
# <m|x^k|n>, origin is 0
#\left\langle m\left|X^{k}\right| n\right\rangle=2^{-\frac{k}{2}} \sqrt{n ! m !}
#\quad \sum_{s=\max \left\{0, \frac{m+n-k}{2}\right\}} \frac{k !}{(m-s) !
# s !(n-s) !(k-m-n+2 s) ! !}
# the problem is that large factorial may meet overflow problem
    assert type(k) is int
    assert type(m) is int
    assert type(n) is int

    if (m+n-k) % 2 == 1:
        return 0
    else:
        factorial = scipy.special.factorial
        factorial2 = scipy.special.factorial2
        s_start = max(0, (m+n-k)//2)
        res =  2**(-k/2) * np.sqrt(float(factorial(m,exact=True))) * \
                np.sqrt(float(factorial(n, exact=True)))
        sum0 = 0.
        for s in range(s_start, min(m,n)+1):
            sum0 +=  factorial(k, exact=True) / factorial(m-s, exact=True) / factorial(s, exact=True) /\
               factorial(n-s, exact=True) / factorial2(k-m-n+2*s, exact=True)

        return res*sum0


def p_power_k(k,m,n):
# <m|p^k|n>
    return x_power_k(k,m,n) * (1j)**(m-n)
