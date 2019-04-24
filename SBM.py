# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
from ephMPS import constant
from ephMPS import classical
from ephMPS.lib import mps as mpslib
import numpy.polynomial.laguerre as La
import numpy.polynomial.legendre as Le
import scipy
import scipy.special

def SBM_init(mol, T=0, pure_Hartree=False, Ehrenfest=False):
    """
    initiate the spin-boson model initial state the spin is alpha and the vibrations
    are on the ground state equilibrium position
    """
    if T == 0:
        if pure_Hartree == False:
            mps_e = np.zeros([1,2,1])
            mps_e[0,0,0] = 1.0
            MPS = [mps_e,]

            for ph in mol[0].ph:
                mps = np.zeros([1,ph.nlevels,1])
                mps[0,0,0] = 1.0
                MPS.append(mps)
            
            WFN = []
            for ph in mol[0].ph_hybrid:
                if Ehrenfest == False:
                    wfn = np.zeros(ph.nlevels)
                    wfn[0] = 1.0
                else:
                    wfn = classical.classical_particle.action_angle(T, ph.omega[0])
                WFN.append(wfn)
            WFN.append(1.0)  
        
        else:
            MPS = []
            wfn_e = np.array([1.0,0.0])
            WFN = [wfn_e,]
            
            for ph in mol[0].ph:
                wfn = np.zeros(ph.nlevels)
                wfn[0] = 1.0
                WFN.append(wfn)

            WFN.append(1.0)  
    
    else:

        beta = constant.T2beta(T)/2.0
        
        def hartree_init(WFN, ph_list):
            for ph in ph_list:
                wfn = np.zeros(ph.nlevels, ph.nlevels)
                
                partition = 0.0
                for il in range(ph.nlevels):
                    pop = np.exp(- beta * il * ph.omega[0])
                    partition += pop**2
                    wfn[il,il] = pop

                WFN.appedn(wfn/np.sqrt(partition))


        if pure_Hartree == False:
            mps_e = np.zeros([1,2,2,1])
            mps_e[0,0,0,0] = 1.0
            MPS = [mps_e,]

            for ph in mol[0].ph:
                mps = np.zeros([1, ph.nlevels, ph.nlevels, 1])
                
                partition = 0.0
                for il in range(ph.nlevels):
                    pop = np.exp(- beta * il * ph.omega[0])
                    partition += pop**2
                    mps[0,il,il,0] = pop

                MPS.append(mps/np.sqrt(partition))
            
            WFN = []
            hartree_init(WFN, mol[0].ph_hybrid)
            WFN.append(1.0)

        else:
            MPS = []
            wfn_e = np.zeros([2,2])
            wfn_e[0,0] = 1.0
            
            WFN = [wfn_e,]
            hartree_init(WFN, mol[0].ph)
            WFN.append(1.0)
            
    
    if pure_Hartree == False:
        MPS = mpslib.MPSdtype_convert(MPS)
    
    if Ehrenfest == False:
        WFN = [wfn.astype(np.complex128) for wfn in WFN[:-1]]+[WFN[-1]]

    return MPS, WFN



class Debye_type_spectral_density_function(object):
    '''
    the Debye-type ohmic spectral density function 
    J(\omega)= \frac{2 \lambda \omega \omega_{c}}{\omega^{2}+\omega_{c}^{2}}
    '''
    def __init__(self, lamb, omega_c):
        self.lamb = lamb
        self.omega_c = omega_c

    def func(self, omega_value):
        '''
        the function of the Debye-type spectral density function
        '''
        return 2.* self.lamb * omega_value*self.omega_c/(omega_value**2 + self.omega_c**2)


class spectral_density_function(object):
    '''
    the ohmic spectral density function
    J(\omega) = \pi / 2 \alpha \omega e^{-\omega/\omega_c}
    '''
    def __init__(self, alpha, omega_c):
        self.alpha = alpha
        self.omega_c = omega_c

    
    def adiabatic_renormalization(self, Delta, p):
        loop = 0
        re = 1.
        while loop<50:
            re_old = re
            omega_l = Delta * re * float(p)
            re = np.exp(-self.alpha*scipy.special.expn(1,omega_l/self.omega_c))
            loop += 1
            print ("adiabatic_renormalization loop,re:",loop, re)
            if np.allclose(re,re_old) == True:
                break
            
        return Delta*re, Delta*re*p

    def func(self, omega_value):
        '''
        the function of the ohmic spectral density function
        '''
        return np.pi / 2. * self.alpha * omega_value * np.exp(-omega_value/self.omega_c)


    def sort(self, omega_value, c_j2, ifsort):
        if ifsort == True:
            idx = np.argsort(c_j2/omega_value)[::-1]
            return omega_value[idx], c_j2[idx]
        else:
            return omega_c, c_j2


    def _dos_Wang1(self, Nb, omega_value):
        '''
        Wang's 1st scheme DOS \rho(\omega)
        '''
        return (Nb+1) / self.omega_c * np.exp(-omega_value/self.omega_c)


    def Wang1(self, Nb, ifsort=True):
        '''
        Wang's 1st scheme discretization
        '''
        omega_value = np.array([-np.log(-float(j)/(Nb+1)+1.)*self.omega_c for j in range(1,Nb+1,1) ])
        
        # general form  
        #c_j2 = 2./np.pi * omega_value * self.func(omega_value) / self._dos_Wang1(Nb, omega_value)
        
        # excat form
        c_j2 = omega_value**2 * self.alpha * self.omega_c / (Nb+1)

        return self.sort(omega_value, c_j2, ifsort)


    def legendre(self, Nb, x0, x1, ifsort=True):
        '''
        Legendre polynomial fit [x0, x1] to [-1,1]
        omega_m is the cutoff
        '''
        omega_value, w = Le.leggauss(Nb)
        omega_value = (omega_value + (x1+x0)/(x1-x0)) * (x1-x0) / 2.
        c_j2 = w * (x1-x0) / 2. * self.alpha * omega_value**2 * np.exp(-omega_value/self.omega_c)
        
        return self.sort(omega_value, c_j2, ifsort)


    def laguerre(self, Nb, ifsort=True):
        assert Nb <= 100

        omega_value, w = La.laggauss(Nb)
        omega_value *= self.omega_c
        c_j2 = w * self.alpha * self.omega_c * omega_value**2
    
        return self.sort(omega_value, c_j2, ifsort)
    

    def trapz(self, Nb, x0, x1, ifsort=True):
        dw = (x1-x0)/float(Nb)
        xlist = [x0+i*dw for i in range(Nb+1)]
        omega_value = np.array([(xlist[i]+xlist[i+1])/2. for i in range(Nb)])
        c_j2 = np.array([(self.func(xlist[i])+self.func(xlist[i+1]))/2 for i in
            range(Nb)]) * 2. / np.pi * omega_value * dw

        return self.sort(omega_value, c_j2, ifsort)
    

    def _opt_cut(self, p):
        '''
        p is the percent of the height of the SDF 
        '''
        assert p < 1 and p > 0
        cut = np.exp(-1.) * p

        def F(x):
            return  x * np.exp(-x) - cut
        def fprime(x):
            return  np.exp(-x)*(1.-x)
        def fprime2(x):
            return  (x-2.)*np.exp(-x)
        
        x1 = scipy.optimize.newton(F, 0.0, fprime=fprime, fprime2=fprime2)
        x2 = scipy.optimize.newton(F, -np.log(cut), fprime=fprime, fprime2=fprime2)

        return x1*self.omega_c, x2*self.omega_c

    
    def plot_data(self, x0, x1, N, omega_value, c_j2, sigma=0.1):
        '''
        plot the spectral density function (continuous and discrete)
        '''
        x = np.linspace(x0,x1,N)
        y_c = self.func(x)
        
        y_d = np.einsum("i,ji -> j", c_j2/omega_value*np.pi/2.* 1 / np.sqrt(2*np.pi*sigma**2),
                np.exp(-(np.subtract.outer(x, omega_value)/sigma)**2/2.))

        return x, y_c, y_d
    
