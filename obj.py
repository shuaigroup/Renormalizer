# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np


class Phonon(object):
    '''
    phonon class has property: 
    frequency : omega{}
    PES displacement: dis
    highest occupation levels: nlevels
    '''
    def __init__(self, omega, displacement, nlevels, force3rd=None, nqboson=1, qbtrunc=0.0):
        # omega is a dictionary for different PES omega[0], omega[1]...
        self.omega = omega
        # dis is a dictionary for different PES dis[0]=0.0, dis[1]...
        self.dis = displacement
        # Hamiltonian of vibration used in TDh
        self.H_vib_dep = None
        self.H_vib_indep = None

        if force3rd is None:
            self.force3rd = {}
            for i in xrange(len(omega)):
                self.force3rd[i] = 0.0
        else:
            self.force3rd = force3rd

        self.nlevels = nlevels
        self.nqboson = nqboson
        self.qbtrunc = qbtrunc
        self.base = int(round(nlevels**(1./nqboson)))

    def printinfo(self):
        print "omega   = ", self.omega
        print "displacement = ", self.dis
        print "nlevels = ", self.nlevels
        print "nqboson = ", self.nqboson
        print "qbtrunc = ", self.qbtrunc
        print "base =", self.base


class Mol(object):
    '''
    molecule class property:
    local excitation energy :  elocalex
    # of phonons : nphs
    condon dipole moment : dipole
    phonon information : ph
    '''
    def __init__(self, elocalex, nphs, dipole, nphs_hybrid=0, Model="Holstein"):
        self.elocalex = elocalex
        self.dipole = dipole
        
        self.nphs = nphs
        self.ph = []
        self.phhop = np.zeros([nphs, nphs])
        self.e0 = 0.0
        
        self.nphs_hybrid = nphs_hybrid
        self.ph_hybrid = []
        self.e0_hybrid = 0.0

        self.Model = Model
        self.Delta = 1.

    def create_phhop(self, phhopmat):
        self.phhop = phhopmat.copy()

    def printinfo(self):
        print "local excitation energy = ", self.elocalex
        print "nphs = ", self.nphs
        print "nphs_hybrid = ", self.nphs_hybrid
        print "dipole = ", self.dipole
    
    def create_ph(self, phinfo, phtype="normal"):
        assert phtype in ["normal","hybrid"]
        if phtype == "normal":
            dst_nphs = self.nphs
            dst_ph = self.ph
        elif phtype == "hybrid":
            dst_nphs = self.nphs_hybrid
            dst_ph = self.ph_hybrid
        
        assert len(phinfo) == dst_nphs
        
        for iph in xrange(dst_nphs):
            ph_local = Phonon(*phinfo[iph])
            dst_ph.append(ph_local) 
        
        # omega*coupling**2: a constant for single mol 
        dst_e0 = 0.0
        for iph in xrange(dst_nphs):
            # only consider two PES
            dst_e0 += 0.5*dst_ph[iph].omega[1]**2 * dst_ph[iph].dis[1]**2 - \
                    dst_ph[iph].dis[1]**3 * dst_ph[iph].force3rd[1]
        
        if phtype == "normal":
            self.e0 = dst_e0 
        elif phtype == "hybrid":
            self.e0_hybrid = dst_e0 

class bidict(dict):
    '''
    bi-dictionary class, doule-way hash table
    '''
    def __init__(self, *args, **kwargs):
        self.inverse = {}
        super(bidict, self).__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super(bidict, self).__setitem__(key, value)
        self.inverse[value] = key

    def __delitem__(self, key):
        del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)
