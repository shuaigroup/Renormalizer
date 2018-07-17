# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

'''
automatic Runge-Kutta method coefficient calculation
'''

import numpy as np

method_list =  ["Forward_Euler","midpoint_RK2","Heun_RK2","Ralston_RK2","Kutta_RK3","C_RK4","38rule_RK4"]


def tableau(RK_method):
    '''
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
    '''
    assert RK_method in method_list

    # print "Runge Kutta method", RK_method

    if RK_method == "Forward_Euler":
        # Euler explicit
        a = np.array([[0]])
        b = np.array([1])
        c = np.array([0])
        Nstage = 1
    elif RK_method in ["midpoint_RK2","Heun_RK2","Ralston_RK2"]:
        if RK_method == "midpoint_RK2":
        # if alpha == 1, midpoint method
            alpha = 1.0
        elif RK_method == "Heun_RK2":
        # if alpha == 0.5, heun's method
            alpha = 0.5
        elif RK_method == "Ralston_RK2":
            alpha = 2.0/3.0

        a = np.array([[0,0],[alpha,0]])
        b = np.array([1-0.5/alpha,0.5/alpha])
        c = np.array([0,alpha])
        Nstage = 2
    elif RK_method == "Kutta_RK3":
        # Kutta's third-order method
        a = np.array([[0,0,0],[0.5,0,0],[-1,2,0]])
        b = np.array([1.0/6.0,2.0/3.0,1.0/6.0])
        c = np.array([0,0.5,1])
        Nstage = 3
    elif RK_method == "C_RK4":
        # Classic fourth-order method
        a = np.array([[0,0,0,0],[0.5,0,0,0],
                      [0,0.5,0,0],[0,0,1,0]])
        b = np.array([1.0/6.0,1.0/3.0,1.0/3.0,1.0/6.0])
        c = np.array([0,0.5,0.5,1.0])
        Nstage = 4
    elif RK_method == "38rule_RK4":
        # 3/8 rule fourth-order method
        a = np.array([[0,0,0,0],\
                      [1.0/3.0,0,0,0],\
                      [-1.0/3.0,1,0,0],\
                      [1,-1,1,0]])
        b = np.array([1.0/8.0, 3.0/8.0, 3.0/8.0, 1.0/8.0])
        c = np.array([0.0, 1.0/3.0, 2.0/3.0 ,1.0])
        Nstage = 4

        a = a.astype(np.float64)
        b = b.astype(np.float64)
        c = c.astype(np.float64)

    return [a,b,c,Nstage]


def coefficient(tableau):
    '''
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
    '''

    a, b, c, Nstage = tableau

    v = np.zeros([Nstage, Nstage+1])  # each row is a point
    k = np.zeros([Nstage, Nstage+1])  # slope at point

    v[0,0] = 1.0

    for ik in range(a.shape[0]):
        k[ik,1:] =  v[ik,:-1]
        k[ik,0] = 0.0

        if ik != a.shape[0]-1:
            v[ik+1,:] = v[0,:]
            for iv in range(a.shape[1]):
                v[ik+1,:] += a[ik+1,iv] * k[iv,:]

    d = v[0,:] + b.dot(k)

    return d


coefficient_dict = {method: coefficient(tableau(method)) for method in method_list}

