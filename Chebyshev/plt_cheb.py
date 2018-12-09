# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import constant



renorm = np.load('cheb_interval.npy')
OMEGA = np.linspace(1.4, 3.0, num=1000) / constant.au2ev
OMEGA = OMEGA[300:680, ]
OMEGA = np.arange(0.1 ,5, 0.001)
#OMEGA = OMEGA / (3/(2*(1-2*0.025))) - (1-2*(0.025))
print('renorm.shaoe', renorm.shape)
for i in range(0, renorm.shape[1]):
    #`renorm[:,i] = OMEGA[i,] * renorm[:,i]
    plt.plot(OMEGA, renorm[:, i]/np.max(renorm[:, i]), label=((i+1)*1000))
plt.legend()
plt.show()
