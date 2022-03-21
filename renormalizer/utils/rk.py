# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

"""
automatic Runge-Kutta method coefficient calculation
"""

import logging

import numpy as np
from scipy.special import factorial

logger = logging.getLogger(__name__)

method_list = [
    "Forward_Euler",
    "midpoint_RK2",
    "Heun_RK2",
    "Ralston_RK2",
    "Kutta_RK3",
    "C_RK4",
    "38rule_RK4",
    "Fehlberg5",
    "RKF45",
    "Cash-Karp45",
]

class TaylorExpansion:
    # taylor expansion of the formal propagator
    # for time-independent Hamiltonian
    def  __init__(self, order):
        self.order = order
        self.coeff = np.array(
                [1.0 / factorial(i) for i in range(self.order + 1)])


class RungeKutta:
    def __init__(self, method="C_RK4"):

        assert method in method_list
        self.method = method

        self.tableau, self.stage, self.order = self.get_tableau()
        
    def get_tableau(self):
        r"""
        Butcher tableau of the explicit Runge-Kutta methods.

        different types of propagation methods: e^-iHdt \Psi
        1.      classical 4th order Runge Kutta
                0   |
                1/2 |  1/2
                1/2 |   0    1/2
                1   |   0     0     1
                ----------------------------
                    |  1/6   1/3   1/3   1/6

        2.      Heun's method
                0   |
                1   |   1
                ----------------------------
                       1/2   1/2
        """

        if self.method == "Forward_Euler":
            # Euler explicit
            a = np.array([[0]])
            b = np.array([1])
            c = np.array([0])
            Nstage = 1
            order = (1,)

        elif self.method in ["midpoint_RK2", "Heun_RK2", "Ralston_RK2"]:
            if self.method == "midpoint_RK2":
                # if alpha == 1, midpoint method
                alpha = 1.0
            elif self.method == "Heun_RK2":
                # if alpha == 0.5, heun's method
                alpha = 0.5
            elif self.method == "Ralston_RK2":
                alpha = 2.0 / 3.0
            else:
                assert False

            a = np.array([[0, 0], [alpha, 0]])
            b = np.array([1 - 0.5 / alpha, 0.5 / alpha])
            c = np.array([0, alpha])
            Nstage = 2
            order = (2,)

        elif self.method == "Kutta_RK3":
            # Kutta's third-order method
            a = np.array([[0, 0, 0], [0.5, 0, 0], [-1, 2, 0]])
            b = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
            c = np.array([0, 0.5, 1])
            Nstage = 3
            order = (3,)

        elif self.method == "C_RK4":
            # Classic fourth-order method
            a = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])
            b = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
            c = np.array([0, 0.5, 0.5, 1.0])
            Nstage = 4
            order = (4,)

        elif self.method == "38rule_RK4":
            # 3/8 rule fourth-order method
            a = np.array(
                [
                    [0, 0, 0, 0],
                    [1.0 / 3.0, 0, 0, 0],
                    [-1.0 / 3.0, 1, 0, 0],
                    [1, -1, 1, 0],
                ]
            )
            b = np.array([1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0])
            c = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
            Nstage = 4
            order = (4,)

        elif self.method == "Fehlberg5":
            a = np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [1 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [3.0 / 32, 9.0 / 32, 0, 0, 0, 0],
                    [1932.0 / 2197, -7200.0 / 2197, 7296.0 / 2197, 0, 0, 0],
                    [439.0 / 216, -8.0, 3680.0 / 513, -845 / 4104, 0.0, 0.0],
                    [-8.0 / 27, 2.0, -3544.0 / 2565, 1859.0 / 4104, -11.0 / 40, 0.0],
                ]
            )
            b = np.array(
                [16.0 / 135, 0.0, 6656.0 / 12825, 28561.0 / 56430, -9.0 / 50, 2.0 / 55]
            )
            c = np.array([0.0, 1.0 / 4, 3.0 / 8, 12.0 / 13, 1.0, 1.0 / 2])
            Nstage = 6
            order = (5,)

        elif self.method == "RKF45":
            a = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0 / 4, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [3.0 / 32, 9.0 / 32, 0.0, 0.0, 0.0, 0.0],
                    [1932.0 / 2197, -7200.0 / 2197, 7296.0 / 2197, 0.0, 0.0, 0.0],
                    [439.0 / 216, -8.0, 3680.0 / 513, -845.0 / 4104, 0.0, 0.0],
                    [-8.0 / 27, 2.0, -3544.0 / 2565, 1859.0 / 4104, -11.0 / 40, 0.0],
                ]
            )
            c = np.array([0.0, 1.0 / 4, 3.0 / 8, 12.0 / 13, 1.0, 1 / 2.0])
            b = np.array(
                [
                    [
                        16.0 / 135,
                        0.0,
                        6656.0 / 12825,
                        28561.0 / 56430,
                        -9.0 / 50,
                        2.0 / 55,
                    ],
                    [25.0 / 216, 0.0, 1408.0 / 2565, 2197.0 / 4104, -1.0 / 5, 0.0],
                ]
            )
            Nstage = 6
            # the order corresponds to b
            order = (5, 4)
        elif self.method == "Cash-Karp45":
            a = np.array(
            [[0, 0, 0, 0, 0, 0],
             [1/5, 0, 0, 0, 0, 0],
             [3/40, 9/40, 0, 0, 0, 0],
             [3/10, -9/10, 6/5, 0, 0, 0],
             [-11/54, 5/2, -70/27, 35/27, 0, 0],
             [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096, 0]])    
            c = np.array([0, 1/5, 3/10, 3/5, 1, 7/8]) 
            b = np.array([
                    [37/378, 0, 250/621, 125/594, 0, 512/1771],
                    [2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4]])
            Nstage = 6
            # the order corresponds to b
            order = (5, 4)
        else:
            assert False

        a = a.astype(np.float64)
        b = b.astype(np.float64).reshape(-1, Nstage)
        c = c.astype(np.float64)

        return [a, b, c], Nstage, order

    def runge_kutta_ti_coefficient(self):
        """
        only suited for time-independent propagator
        y'(t) = fy(t) f is time-independent
        the final formula is
        y(t+dt) = d0 y(t) + d1 fy(t) dt + d2 f^2 y(t) dt^2 + ...
            0  f  f^2 f^3 f^4
        v0
        v1
        v2
        v3
        Though, each order has different versions of RK methods, if f is time
        independent, the coefficient is the same. For example, Classical 4th order
        Runge Kutta and 3/8 rule Runge Kutta has some coefficient.
        """

        a, b, c = self.tableau
        Nstage = self.stage

        table = np.zeros([Nstage + 1, Nstage + 1])
        table[0, 0] = 1.0
        for istage in range(Nstage):
            table[istage + 1, 2:] = a[istage, :].dot(table[1:, 1:])[:-1]
            table[istage + 1, 1] = 1.0

        if b.ndim == 1:
            # before RK4
            coeff = np.zeros(Nstage + 1)
            coeff[0] = 1.0
            coeff[1:] = b.dot(table[1:, 1:])
        else:
            # after RK4
            coeff = np.zeros((b.shape[0], Nstage + 1))
            coeff[:, 0] = 1.0
            coeff[:, 1:] = b.dot(table[1:, 1:])

        # actully it is Taylor expansion for time independent f

        return coeff
