# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
from ephMPS import constant

class classical_particle(object):
    
    def __init__(self, p,q):

        self.p, self.q = p, q
        self.qlast = None

    @classmethod
    def action_angle(cls, T, omega, delta=0.):
        '''
        q = \sqrt{2n+1} sin x + delta
        p = \sqrt{2n+1} cos x
        x is a random variable in [0, 2\pi],
        delta is the  initial
        mean position of the bath modes
        P(n) ~ exp(-\beta n omega)
        '''
        x = np.random.random(1) * np.pi * 2
        if T == 0:
            n = 0
        else:
            beta = constant.T2beta(T)
            random = np.random.random(1)
            
            n = 0
            while True:
                if random >= (1.-np.exp(-n*omega*beta))/(1.-np.exp(-omega*beta)) \
                    and random < (1.-np.exp(-(n+1)*omega*beta))/(1.-np.exp(-omega*beta)):
                    break
                n += 1

        q = (np.sqrt(2.*n+1.0) * np.sin(x) + delta) / np.sqrt(omega)
        p = (np.sqrt(2.*n+1.0) * np.cos(x)) * np.sqrt(omega)
        
        return cls(p, q)
    
    @classmethod
    def Wigner_sampling(cls):
        pass

    
