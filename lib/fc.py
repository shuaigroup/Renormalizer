# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

'''
Franck-Condon Integral
<\Psi_1 v_1| \Psi_1 v_2>
'''

from scipy.misc import factorial
import numpy as np


def FC_integral_dho(w1, w2, v1, v2, Q):
    '''
    Franck-Condon Integral between
    vibratioanl wavefunction of displaced hamonic oscillator
    See Chih-Kai Lin, Phys. Chem. Chem. Phys., 2010, 12, 11432–11444
    '''
    def Pr(x):
        if x%2 == 0:
            return 0
        else:
            return 1
    
    beta1 = w1 
    beta2 = w2

    
    coeff1 = 1.0 /(beta1 + beta2) 
    coeff2 = (beta1-beta2)/(beta1+beta2)

    fc = 0.0
    for l in xrange(min(v1,v2)+1):
        for m in xrange(v1-l+1):
            for n in xrange(v2-l+1):
                pr1 = Pr(v1-l-m-1)
                pr2 = Pr(v2-l-n-1)
                if pr1 == 0 or pr2 == 0:
                    continue
                else:
                    fc += pr1 * pr2 * (-1.)**((v2-l+n)/2.) * \
                        2.**(2.*l+m+n)/factorial(l)/factorial(m)/factorial(n) * \
                        coeff1**(l+m+n) * beta1**((l+m)/2.+n) * \
                        beta2**((l+n)/2.+m) * \
                        Q**(m+n) * coeff2**((v1+v2-2.*l-m-n)/2.) / \
                        factorial((v1-l-m)/2.) / factorial((v2-l-n)/2.)
    
    fc *= (factorial(v1)*factorial(v2) / (2.**(v1+v2)) * \
            2.*np.sqrt(beta1*beta2)/(beta1+beta2))**0.5 *\
            np.exp(-beta1*beta2*Q**2./2./(beta1+beta2))


    return fc
