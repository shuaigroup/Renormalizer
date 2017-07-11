#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Phonon(object):

    def __init__(self, omega, ephcoup, nlevels):
        self.omega = omega
        self.ephcoup = ephcoup
        self.nlevels = nlevels

    def printinfo(self):
        print "omega   = ", self.omega
        print "ephcoup = ", self.ephcoup
        print "nlevels = ", self.nlevels


class Mol(object):

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
    def __init__(self, *args, **kwargs):
        self.inverse = {}
        super(bidict, self).__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super(bidict, self).__setitem__(key, value)
        self.inverse[value] = key

    def __delitem__(self, key):
        del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)
