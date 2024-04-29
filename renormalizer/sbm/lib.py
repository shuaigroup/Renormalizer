# -*- coding: utf-8 -*-

import logging
import numpy as np
import numpy.polynomial.laguerre as la
import numpy.polynomial.legendre as le
import scipy
import scipy.special
import scipy.optimize

from renormalizer.model import Phonon, SpinBosonModel
from renormalizer.utils import Quantity


logger = logging.getLogger(__name__)


class DebyeSpectralDensityFunction:
    r"""
    the Debye-type ohmic spectral density function
    J(\omega)= \frac{2 \lambda \omega \omega_{c}}{\omega^{2}+\omega_{c}^{2}}
    """

    def __init__(self, lamb, omega_c):
        self.lamb = lamb
        self.omega_c = omega_c

    def func(self, omega_value):
        """
        the function of the Debye-type spectral density function
        """
        return 2. * self.lamb * omega_value * self.omega_c / (omega_value ** 2 + self.omega_c ** 2)


DebyeSDF = DebyeSpectralDensityFunction


class SpectralDensityFunction:
    r"""
    the ohmic spectral density function
    J(\omega) = \pi / 2 \alpha \omega e^{-\omega/\omega_c}
    """

    def __init__(self, alpha, omega_c: Quantity):
        self.alpha = alpha
        self.omega_c = omega_c.as_au()

    def adiabatic_renormalization(self, delta_quan: Quantity, p: float):
        delta = delta_quan.as_au()
        loop = 0
        re = 1.
        while loop < 50:
            re_old = re
            omega_l = delta * re * p
            re = np.exp(-self.alpha * scipy.special.expn(1, omega_l / self.omega_c))
            loop += 1
            if np.allclose(re, re_old):
                break

        return delta_quan * re, Quantity(delta * re * p)

    def func(self, omega_value):
        """
        the function of the ohmic spectral density function
        """
        return np.pi / 2. * self.alpha * omega_value * np.exp(-omega_value / self.omega_c)

    @staticmethod
    def post_process(omega_value, c_j2, ifsort):
        displacement_array = np.sqrt(c_j2) / omega_value ** 2
        if ifsort:
            idx = np.argsort(c_j2 / omega_value)[::-1]
        else:
            idx = np.arange(len(omega_value))
        omega_list = []
        displacement_list = []
        for i in idx:
            omega_list.append(Quantity(omega_value[i]))
            displacement_list.append(Quantity(displacement_array[i]))
        return omega_list, displacement_list

    def _dos_Wang1(self, nb, omega_value):
        r"""
        Wang's 1st scheme DOS \rho(\omega)
        """
        return (nb + 1) / self.omega_c * np.exp(-omega_value / self.omega_c)

    def Wang1(self, nb, ifsort=True):
        """
        Wang's 1st scheme discretization
        """
        omega_value = np.array([-np.log(-float(j) / (nb + 1) + 1.) * self.omega_c for j in range(1, nb + 1, 1)])

        # general form
        # c_j2 = 2./np.pi * omega_value * self.func(omega_value) / self._dos_Wang1(nb, omega_value)

        # excat form
        c_j2 = omega_value ** 2 * self.alpha * self.omega_c / (nb + 1)

        return self.post_process(omega_value, c_j2, ifsort)

    def legendre(self, nb, x0, x1, ifsort=True):
        """
        Legendre polynomial fit [x0, x1] to [-1,1]
        omega_m is the cutoff
        """
        omega_value, w = le.leggauss(nb)
        omega_value = (omega_value + (x1 + x0) / (x1 - x0)) * (x1 - x0) / 2.
        c_j2 = w * (x1 - x0) / 2. * self.alpha * omega_value ** 2 * np.exp(-omega_value / self.omega_c)

        return self.post_process(omega_value, c_j2, ifsort)

    def laguerre(self, nb, ifsort=True):
        assert nb <= 100

        omega_value, w = la.laggauss(nb)
        omega_value *= self.omega_c
        c_j2 = w * self.alpha * self.omega_c * omega_value ** 2

        return self.post_process(omega_value, c_j2, ifsort)

    def trapz(self, nb, x0, x1, ifsort=True):
        dw = (x1 - x0) / float(nb)
        xlist = [x0 + i * dw for i in range(nb + 1)]
        omega_value = np.array([(xlist[i] + xlist[i + 1]) / 2. for i in range(nb)])
        c_j2 = np.array([(self.func(xlist[i]) + self.func(xlist[i + 1])) / 2 for i in range(nb)]) * 2. / np.pi * omega_value * dw

        return self.post_process(omega_value, c_j2, ifsort)

    def _opt_cut(self, p):
        """
        p is the percent of the height of the SDF
        """
        assert 0 < p < 1
        cut = np.exp(-1.) * p

        def F(x):
            return x * np.exp(-x) - cut

        def fprime(x):
            return np.exp(-x) * (1. - x)

        def fprime2(x):
            return (x - 2.) * np.exp(-x)

        x1 = scipy.optimize.newton(F, 0.0, fprime=fprime, fprime2=fprime2)
        x2 = scipy.optimize.newton(F, -np.log(cut), fprime=fprime, fprime2=fprime2)

        return x1 * self.omega_c, x2 * self.omega_c

    def plot_data(self, x0, x1, n, omega_value, c_j2, sigma=0.1):
        """
        plot the spectral density function (continuous and discrete)
        """
        x = np.linspace(x0, x1, n)
        y_c = self.func(x)

        y_d = np.einsum("i,ji -> j", c_j2 / omega_value * np.pi / 2. * 1 / np.sqrt(2 * np.pi * sigma ** 2),
                        np.exp(-(np.subtract.outer(x, omega_value) / sigma) ** 2 / 2.))

        return x, y_c, y_d


OhmicSDF = SpectralDensityFunction


class ColeDavidsonSDF:
    """
    the ColeDavidson spectral density function
    """

    def __init__(self, ita, omega_c, beta, omega_limit):
        self.ita = ita
        self.omega_c = omega_c
        self.beta = beta
        self.omega_limit = omega_limit


    def reno(self, omega_l):
        def integrate_func(x):
            return self.func(x) / x**2

        res = scipy.integrate.quad(integrate_func, a=omega_l,
                b=omega_l*1000)
        logger.info(f"integrate: {res[0]}, {res[1]}")
        re = np.exp(-res[0]*2/np.pi)

        return re


    def func(self, omega_value):
        """
        the function of the spectral density function
        """
        theta = np.arctan(omega_value/self.omega_c)
        return self.ita * np.sin(self.beta * theta) / (1 + omega_value**2/self.omega_c**2) ** (self.beta / 2)


    def _dos_Wang1(self, A, omega_value):
        """
        Wang's 1st scheme DOS \rho(\omega)
        """

        return A * self.func(omega_value) / omega_value

    def Wang1(self, nb):
        """
        Wang's 1st scheme discretization
        """
        def integrate_func(x):
            return self.func(x) / x
        A = (nb + 1 ) / scipy.integrate.quad(integrate_func, a=0, b=self.omega_limit)[0]
        logger.info(scipy.integrate.quad(integrate_func, a=0, b=self.omega_limit)[0] * 4 / np.pi)
        logger.info(2*self.ita)
        nsamples = int(1e7)
        delta = self.omega_limit / nsamples
        omega_value_big = np.linspace(delta, self.omega_limit, nsamples)
        dos = self._dos_Wang1(A, omega_value_big)
        rho_cumint = np.cumsum(dos) * delta
        diff = (rho_cumint % 1)[1:] - (rho_cumint % 1)[:-1]
        idx = np.where(diff < 0)[0]
        omega_value = omega_value_big[idx]
        logger.info(len(omega_value))
        assert len(omega_value) == nb

        # general form
        c_j2 = 2./np.pi * omega_value * self.func(omega_value) / self._dos_Wang1(A, omega_value)


        return omega_value, c_j2


def param2mollist(alpha: float, raw_delta: Quantity, omega_c: Quantity, renormalization_p: float, n_phonons: int):
    sdf = SpectralDensityFunction(alpha, omega_c)
    delta, max_omega = sdf.adiabatic_renormalization(raw_delta, renormalization_p)
    omega_list, displacement_list = sdf.trapz(n_phonons, 0.0, max_omega.as_au())

    ph_list = [Phonon.simplest_phonon(o, d) for o,d in zip(omega_list, displacement_list)]
    return SpinBosonModel(Quantity(0), delta, ph_list)
