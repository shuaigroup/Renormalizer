# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

'''
discretization continuous spectral density to discrete electron-phonon coupling
1. direct discretization 
2. polynomial discretization
Phys. Rev. B 92, 155126
'''

import numpy as np
from scipy.integrate import quad
import scipy.linalg


def discrete_Legendre(num, slope_m, intercept_c):
    '''
    See Numerical Recipes  "Case of Known Recurrences"
    Legendre polynomials range [-1,1] w(x)=1
    recurrence relation
    P_{n+1} = x P_n - n**2/(4n**2-1) P_{n-1}
    
    The shifted polynomial 
    y = mx+c
    alpha' = m*alpha + c
    beta'  = m**2 beta
    
    from [-1,1] to [0, omega_c]
    m = omega_c/2
    c = omega_c/2
    '''
    
    Alpha = np.zeros(num)
    Beta = np.zeros(num)
    for i in xrange(num):
        Alpha[i] = intercept_c
        Beta[i] = float(i**2)/float(4*i**2-1) * slope_m**2
    
    M = np.diag(Alpha) + np.diag(np.sqrt(Beta[1:]),k=1) \
            + np.diag(np.sqrt(Beta[1:]),k=-1)
    
    # w is the optimal position
    # v**2 is the weight, remember if the polynomial is not normalized, the
    # weight should multiply int_a^b w(x)
    w, v = scipy.linalg.eigh(M)    
    
    return w, v[0,:]**2


def spectral_density_Debye():
    pass


def spectral_density_Lorentzian(omega, eta, gamma, omega_v, mode=1, omega2=None):
    '''
    Lorentzian type spectral density
    J(omega) = 2*eta*gamma*(omega_v^2+gamma^2)*omega /  
        [(omega+omega_v)^2+gamma^2] / [(omega-omega_v)^2+gamma^2]
    J. Chem. Phys. 143, 064109 (2015)
    and analytical integral
    1  J(omega)
    2. int_{omega_1}^{omega_2} J(omega) domega 
    3. int_{omega_1}^{omega_2} J(omega)*omega domega 
    4. int_{omega_1}^{omega_2} J(omega)/omega domega 
    '''
    def Jdw(x, a, b): 
        return (np.arctan((-a+x)/b)-np.arctan((a+x)/b))/4./a/b
    
    def Jwdw(x, a, b):
        return (np.arctan((-a+x)/b)+np.arctan((a+x)/b))/4./b +  \
              (np.log(a**2+b**2-2.*a*x+x**2) - np.log(a**2+b**2+2.*a*x+x**2)) \
              /8./a
    
    def J_wdw(x, a, b):
        return -(2.*a*np.arctan((a-x)/b) - 2.*a*np.arctan((a+x)/b) +  \
              b*np.log(a**2+b**2-2.*a*x+x**2) - b*np.log(a**2+b**2+2.*a*x+x**2)) \
              / (8.*a**3*b + 8*a*b**3)
    
    if mode == 1:
        out = 2.*eta*gamma*(omega_v**2+gamma**2)*omega \
            /((omega+omega_v)**2+gamma**2)/((omega-omega_v)**2+gamma**2)
    else:
        if mode == 2:
            up = Jdw(omega2, omega_v, gamma)
            down = Jdw(omega, omega_v, gamma)
        elif mode == 3:
            up = Jwdw(omega2, omega_v, gamma)
            down = Jwdw(omega, omega_v, gamma)
        elif mode == 4:
            up = J_wdw(omega2, omega_v, gamma)
            down = J_wdw(omega, omega_v, gamma)
        out = (up-down)*2.*eta*gamma*(omega_v**2+gamma**2)
    
    return out


def discrete_mean(spectral_func, args, left, right, nx):
    '''
    mean method to discrete the spectral funciton
    int dw w J(w) / int dw J(w)
    Phys. Rev. B 92, 155126
    '''
    xlist = [left,right]
    num = 0
    while True:
        addlist = []
        for ix in xrange(1,len(xlist)):
            intJ = spectral_func(xlist[ix-1], *args, mode=2, omega2=xlist[ix])
            intJw = spectral_func(xlist[ix-1], *args, mode=3, omega2=xlist[ix])
            addlist.append(intJw/intJ)
            num += 1
            if num == nx:
                break
        xlist += addlist
        xlist = sorted(xlist)
        if num == nx:
            break
    
    VVlist = []
    for ix in xrange(1,len(xlist)-1):
        lx = (xlist[ix-1] + xlist[ix]) / 2.
        rx = (xlist[ix] + xlist[ix+1]) / 2.
        if ix == 1:
            lx = left
        if ix == len(xlist)-2:
            rx = right
        VV = spectral_func(lx, *args, mode=2, omega2=rx)
        VVlist.append(VV)
    
    xpos = np.array(xlist[1:-1])
    ephcoup = np.sqrt(np.array(VVlist)/np.pi)/xpos

    return xpos, ephcoup

