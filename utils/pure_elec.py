# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

'''
pure electronic exciton
nearest neighbour interaction
'''
import itertools
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

periodic = True
dim1 = 4
dim2 = 4
dim3 = 1
Nsites =  2
# interaction term 0: on unit, 1: dim1, 2: dim2, 3: dim3

V = np.zeros([Nsites, Nsites, 4])
V[0,1,0] = 114.8
V[1,0,0] = 114.8
V[0,1,1] = 114.8
V[0,0,2] = 82.6
V[1,1,2] = 82.6

#V[0,0,1] = 114.8
#def pure_electronic(V):
H = np.zeros([Nsites,dim1,dim2,dim3,Nsites,dim1,dim2,dim3])

if periodic == False:
    condition_dim1 = [[1,0,0]]
    condition_dim2 = [[0,1,0]]
    condition_dim3 = [[0,0,1]]
else:
    condition_dim1 = [[1,0,0], [1-dim1,0,0]]
    condition_dim2 = [[0,1,0], [0,1-dim2,0]]
    condition_dim3 = [[0,0,1], [0,0,1-dim3]]

for idim1, idim2, idim3, jdim1, jdim2, jdim3 in \
    itertools.product(range(dim1),range(dim2),range(dim3), range(dim1),\
            range(dim2), range(dim3)):
        # i is the next mol, j is the first mol
        vector = [idim1-jdim1, idim2-jdim2, idim3-jdim3]
        
        if vector in [[0,0,0]]:
            V_dim = 0
        elif vector in condition_dim1:
            V_dim = 1
        elif vector in condition_dim2:
            V_dim = 2
        elif vector in condition_dim3:
            V_dim = 3
        else:
            continue

        for isite, jsite in itertools.product(range(Nsites), range(Nsites)):
            H[isite, idim1, idim2, idim3, jsite, jdim1, jdim2, jdim3] = V[isite, jsite, V_dim]
            H[jsite, jdim1, jdim2, jdim3, isite, idim1, idim2, idim3] = V[isite, jsite, V_dim]

w, v = scipy.linalg.eigh(H.reshape(Nsites*dim1*dim2*dim3,Nsites*dim1*dim2*dim3))
print w

transdip = np.sum(v, axis=0)
plt.plot(w, transdip**2)
plt.show()


