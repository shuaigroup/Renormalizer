# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>


class Phonon(object):
    '''
    phonon class has property: 
    frequency : omega
    electron-phonon coupling : ephcoup
    highest occupation levels: nlevels
    '''
    def __init__(self, omega, ephcoup, nlevels, nqboson=1, qbtrunc=0.0):
        self.omega = omega
        self.ephcoup = ephcoup
        self.nlevels = nlevels
        self.nqboson = nqboson
        self.qbtrunc = qbtrunc

    def printinfo(self):
        print "omega   = ", self.omega
        print "ephcoup = ", self.ephcoup
        print "nlevels = ", self.nlevels
        print "nqboson = ", self.nqboson
        print "qbtrunc = ", self.qbtrunc

class Mol(object):
    '''
    molecule class property:
    local excitation energy :  elocalex
    # of phonons : nphs
    condon dipole moment : dipole
    phonon information : ph
    '''
    def __init__(self, elocalex, nphs, dipole):
        self.elocalex = elocalex
        self.nphs = nphs
        self.dipole = dipole
        self.ph = []

    def printinfo(self):
        print "local excitation energy = ", self.elocalex
        print "nphs = ", self.nphs
        print "dipole = ", self.dipole
    
    def create_ph(self, phinfo):
        assert len(phinfo) == self.nphs

        for iph in xrange(self.nphs):
            ph_local = Phonon(*phinfo[iph])
            self.ph.append(ph_local) 


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
