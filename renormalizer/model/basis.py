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
        return f"(dof: {self.dof}, nbas: {self.nbas}, qn: {self.sigmaqn})"

    def op_mat(self, op: Op):
        """
        Matrix representation under the basis set of the input operator.
        The factor is included.

        Parameters
        ----------
        op : Op
            The operator. For basis set with only one DoF, :class:``str`` is also acceptable.

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
        return f"(dof: {self.dof}, x0: {self.x0}, omega: {self.omega}, nbas: {self.nbas})"

    def op_mat(self, op: Union[Op, str]):
        if not isinstance(op, Op):
            op = Op(op, None)
        op_symbol, op_factor = op.symbol, op.factor

        if op_symbol in ["b", "b b", r"b^\dagger", r"b^\dagger b^\dagger", r"b^\dagger b", r"b b^\dagger", r"b^\dagger+b"]:
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

        elif op_symbol == r"b^\dagger+b":
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
                mat = np.sqrt(0.5/self.omega) * self.op_mat(r"b^\dagger+b") + np.eye(self.nbas) * self.x0
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
                mat += 2 * self.x0 * np.sqrt(0.5/self.omega) * self.op_mat(r"b^\dagger+b")

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

class BasisHopsBoson(BasisSet):
    r"""
    Bosonic like basis but with uncommon ladder operator, used in Hierarchy of Pure States method

    .. math::
        \tilde{b}^\dagger | n \rangle = (n+1) | n+1\rangle \\
        \tilde{b} | n \rangle = | n-1\rangle

    Parameters
    ----------
    dof :
        The name of the DoF contained in the basis set. The type could be anything that can be hashed.
    nbas : int
        number of dimension of the basis set (highest occupation number)

    """

    is_phonon = True

    def __init__(self, dof, nbas):
        super().__init__(dof, nbas, [0] * nbas)

    def op_mat(self, op: Union[Op, str]):
        if not isinstance(op, Op):
            op = Op(op, None)
        op_symbol, op_factor = op.symbol, op.factor

        if op_symbol == r"b^\dagger b":
            mat = np.diag(np.arange(self.nbas))
        elif op_symbol == r"\tilde{b}^\dagger":
            #\tilde{b}^\dagger |n\rangle = n+1 |n+1 \rangle
            mat = np.diag(np.arange(1, self.nbas), k=-1)
        elif op_symbol == r"\tilde{b}":
            #\tilde{b} |n\rangle = |n-1 \rangle
            mat = np.diag(np.ones(self.nbas-1), k=1)
        elif op_symbol == "I":
            mat = np.eye(self.nbas)
        else:
            raise ValueError(f"op_symbol:{op_symbol} is not supported.")
        return mat * op_factor

    def copy(self, new_dof):
        return self.__class__(new_dof, self.nbas)


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

    Operators supported:
        .. math::
            I, x, x^1, x^2, x^\textrm{moment}, partialx, partialx^2, p, p^2,
            x partialx, x^2 p^2, x^2 partialx, x p^2, x^3 p^2

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

        assert xi < xf
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

        elif op_symbol.split("^")[0] == "x" and " " not in op_symbol:
            if len(op_symbol.split("^")) == 1:
                # legacy for check
                mat1 = np.zeros((self.nbas, self.nbas))
                for j in range(1,self.nbas+1,1):
                    for k in range(1,self.nbas+1,1):
                        a1 = (j+k)*np.pi/self.L
                        a2 = (j-k)*np.pi/self.L
                        if (j+k)%2 == 1:
                            res = -1/self.L*(-2/a1**2+2/a2**2)
                        elif j-k == 0:
                            res = self.xi + 0.5*self.L
                        else:
                            res = 0
                        mat1[j-1,k-1] = res

                mat = self._I()*self.xi+self._u()
                assert np.allclose(mat, mat1)

                mat = self.dvr_v.T @ mat @ self.dvr_v
            else:
                moment = float(op_symbol.split("^")[1])
                if moment == 1:
                    mat = self.op_mat("x")
                if moment == 2:
                    mat = self._I()*self.xi**2+self._u()*self.xi*2+self._uu()
                    mat = self.dvr_v.T @ mat @ self.dvr_v
                else:
                    mat = np.diag(self.dvr_x ** moment)

        elif set(op_symbol.split(" ")) == set("x"):
            moment = len(op_symbol.split(" "))
            mat = self.op_mat(f"x^{moment}")

        elif op_symbol == "partialx":
            # legacy for check
            mat1 = np.zeros((self.nbas, self.nbas))
            for j in range(self.nbas):
                for k in range(j):
                    if (j-k) % 2 != 0:
                        mat1[j,k] = 4 / self.L * (j+1) * (k+1) / ((j+1)**2 - (k+1)**2)
            mat1 -= mat1.T

            mat = self._du()
            assert np.allclose(mat, mat1)

            mat = self.dvr_v.T @ mat @ self.dvr_v

        elif op_symbol in ["partialx^2", "partialx partialx"]:
            mat = self.op_mat("p^2") * -1

        elif op_symbol == "p":
            mat = self.op_mat("partialx") * -1.0j

        elif op_symbol == "p^2":
            # legacy for check
            mat1 = np.diag(np.arange(1, self.nbas+1)*np.pi/self.L)**2
            mat = np.einsum("jk,k->jk",self._I(),self._eigene()*2)
            assert np.allclose(mat, mat1)
            mat = self.dvr_v.T @ mat @ self.dvr_v

        elif op_symbol[:3] == "cos":
            # cos(alpha * x), the format cos(x,float)
            term_split = op_symbol[4:-1].split(",")
            if "j" in term_split[1]:
                scalar = complex(term_split[1])
            else:
                scalar = float(term_split[1])
            mat = np.diag(np.cos(np.diag(self.op_mat(term_split[0]))*scalar))

        elif op_symbol == "x partialx":
            # legacy for check
            mat1 = np.zeros((self.nbas, self.nbas))
            for j in range(1,self.nbas+1,1):
                for k in range(1,self.nbas+1,1):
                    a1 = (j+k)*np.pi/self.L
                    a2 = (j-k)*np.pi/self.L
                    if (j+k)%2 == 1:
                        res = k*np.pi/self.L**2*(self.xi*(1/a1+1/a2)*2 +
                                self.L*(1/a1+1/a2))
                    elif j-k == 0:
                        res = -k*np.pi/self.L*(1/a1)
                    else:
                        res = -k*np.pi/self.L*(1/a1+1/a2)
                    mat1[j-1,k-1] = res
            mat = self._du()*self.xi + self._udu()
            assert np.allclose(mat, mat1)
            mat = self.dvr_v.T @ mat @ self.dvr_v

        elif op_symbol == "x^2 p^2":

            # legacy for check
            mat1 = np.zeros((self.nbas, self.nbas))
            # analytical integral
            for j in range(1,self.nbas+1):
                for k in range(1,self.nbas+1):

                    a1 = (j-k)*np.pi/self.L
                    a2 = (j+k)*np.pi/self.L

                    if (j+k)%2 == 1:
                        res = 2*self.xi/self.L*2*(1/a2**2-1/a1**2) + 2*(1/a2**2-1/a1**2)
                    elif j-k == 0:
                        res = self.xi**2 + self.xi*self.L + 1/3*self.L**2 - 2/a2**2
                    else:
                        res = 2*(1/a1**2-1/a2**2)
                    mat1[j-1,k-1] = res * k**2*np.pi**2/self.L**2

            tmp = self._I()*self.xi**2 + self._u()*2*self.xi + self._uu()
            mat = np.einsum("jk,k->jk", tmp, self._eigene()*2)
            assert np.allclose(mat, mat1)
            mat = self.dvr_v.T @ mat @ self.dvr_v

        elif op_symbol == "x^2 partialx^2":
            mat = self.op_mat("x^2 p^2") * -1

        elif op_symbol == "x^2 partialx":

            mat = self._uudu() + 2*self.xi*self._udu() + self.xi**2*self._du()
            mat = self.dvr_v.T @ mat @ self.dvr_v

        elif op_symbol == "x p^2":
            ## p^2 is 2H
            # legacy for check
            mat1 = self.dvr_v @ self.op_mat("x") @ self.dvr_v.T
            mat1 = np.einsum("jk,k->jk",mat1,
                    np.arange(1,self.nbas+1)**2*np.pi**2/self.L**2)

            mat = np.einsum("jk, k-> jk", self._I()*self.xi + self._u(), self._eigene()*2)
            assert np.allclose(mat, mat1)
            mat = self.dvr_v.T @ mat @ self.dvr_v

        elif op_symbol == "x partialx^2":
            mat = self.op_mat("x p^2") * -1

        elif op_symbol == "x^3 p^2":
            tmp = self._I()*self.xi**3 + 3*self._uu()*self.xi + 3*self._u()*self.xi**2 + self._uuu()
            mat = np.einsum("jk,k->jk", tmp, self._eigene()*2)
            mat = self.dvr_v.T @ mat @ self.dvr_v

        elif op_symbol == "x^3 partialx^2":
            mat = self.op_mat("x^3 p^2") * -1
        else:
            raise ValueError(f"op_symbol:{op_symbol} is not supported. ")

        return mat * op_factor

    def _du(self):
        # int_0^L <j(u)|1*du|k(u)>  u=x-xi du=\frac{\partial}{\partial u}
        mat = np.zeros((self.nbas, self.nbas))
        for j in range(1, self.nbas+1):
            for k in range(1, self.nbas+1):
                if (j+k)%2 == 1:
                    mat[j-1,k-1] = 4*k*j/self.L/(j**2-k**2)
        return mat

    def _udu(self):
        # int_0^L <j(u)|u*du|k(u)>
        mat = np.zeros((self.nbas, self.nbas))
        for j in range(1, self.nbas+1):
            for k in range(1, self.nbas+1):
                a1 = (j+k)*np.pi/self.L
                a2 = (j-k)*np.pi/self.L
                if (j+k)%2 == 1:
                    res = self.L/a1 + self.L/a2
                elif j == k:
                    res = -self.L/a1
                else:
                    res = -self.L/a1 - self.L/a2
                mat[j-1,k-1] = k*np.pi/self.L**2*res
        return mat

    def _uudu(self):
        # int_0^L <j(u)|u^2*du|k(u)>
        mat = np.zeros((self.nbas, self.nbas))
        for j in range(1, self.nbas+1):
            for k in range(1, self.nbas+1):
                a1 = (j+k)*np.pi/self.L
                a2 = (j-k)*np.pi/self.L
                if (j+k)%2 == 1:
                    res = -4/a1**3 + self.L**2/a1 - 4/a2**3 + self.L**2/a2
                elif j == k:
                    res = -self.L**2/a1
                else:
                    res = -self.L**2/a1 - self.L**2/a2
                mat[j-1,k-1] = k*np.pi/self.L**2*res
        return mat

    def _I(self):
        # int_0^L <j(u)|1|k(u)>
        mat = np.eye(self.nbas)
        return mat

    def _u(self):
        # int_0^L <j(u)|u|k(u)>
        mat = np.zeros((self.nbas, self.nbas))
        for j in range(1,self.nbas+1,1):
            for k in range(1,self.nbas+1,1):
                a1 = (j+k)*np.pi/self.L
                a2 = (j-k)*np.pi/self.L
                if (j+k)%2 == 1:
                    res = -2/a1**2+2/a2**2
                elif j-k == 0:
                    res = -0.5*self.L**2
                else:
                    res = 0
                mat[j-1,k-1] = -1/self.L*res
        return mat

    def _uu(self):
        # int_0^L <j(u)|uu|k(u)>
        mat = np.zeros((self.nbas, self.nbas))
        for j in range(1,self.nbas+1,1):
            for k in range(1,self.nbas+1,1):
                a1 = (j+k)*np.pi/self.L
                a2 = (j-k)*np.pi/self.L
                if (j+k)%2 == 1:
                    res = 2*self.L*(-1/a1**2+1/a2**2)
                elif j-k == 0:
                    res = 2*self.L/a1**2 - 1/3*self.L**3
                else:
                    res = 2*self.L*(1/a1**2 - 1/a2**2)
                mat[j-1,k-1] = -1/self.L*res
        return mat

    def _uuu(self):
        # int_0^L <j(u)|uuu|k(u)>
        mat = np.zeros((self.nbas, self.nbas))
        for j in range(1,self.nbas+1,1):
            for k in range(1,self.nbas+1,1):
                a1 = (j+k)*np.pi/self.L
                a2 = (j-k)*np.pi/self.L
                if (j+k)%2 == 1:
                    res = -3*self.L**2/a1**2 + 12/a1**4 + 3*self.L**2/a2**2 - 12/a2**4
                elif j-k == 0:
                    res = 3*self.L**2/a1**2 - self.L**4/4
                else:
                    res = 3*self.L**2/a1**2 - 3*self.L**2/a2**2
                mat[j-1,k-1] = -1/self.L*res
        return mat

    def _eigene(self):
        return np.pi**2*np.arange(1,self.nbas+1)**2/self.L**2/2

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
            if op_symbol1 == "I" and op_symbol2 == "I":
                return np.eye(self.nbas)
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
            if op_symbol1 == "I" and op_symbol2 == "I":
                return np.eye(self.nbas)
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
            if op_symbol.count("I") == len(op_symbol):
                return np.eye(self.nbas)
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

    Examples
    --------
    >>> b = BasisHalfSpin(0)
    >>> b.op_mat("X")
    array([[0., 1.],
           [1., 0.]])
    >>> -1 * b.op_mat("iY") @ b.op_mat("iY")  # convenient for real Hamiltonian
    array([[1., 0.],
           [0., 1.]])
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
            if op_symbol == "I":
                mat = np.eye(2)
            elif op_symbol in ["sigma_x", "X", "x"]:
                mat = np.diag([1.], k=1)
                mat = mat + mat.T.conj()
            elif op_symbol in ["sigma_y", "Y", "y"]:
                mat = np.diag([-1.0j], k=1)
                mat = mat + mat.T.conj()
            elif op_symbol in ["isigma_y", "iY", "iy"]:
                mat = (1j * self.op_mat("Y")).real
            elif op_symbol in ["sigma_z", "Z", "z"]:
                mat = np.diag([1.,-1.], k=0)
            elif op_symbol in ["sigma_-", "-"]:
                mat = np.diag([1.], k=-1)
            elif op_symbol in ["sigma_+", "+"]:
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
