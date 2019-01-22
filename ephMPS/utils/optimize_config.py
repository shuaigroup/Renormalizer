# -*- coding: utf-8 -*-

class OptimizeConfig:

    def __init__(self):
        self.procedure = [[10,0.4],[20,0.2],[30,0.1],[40,0],[40,0]]
        self.method = "2site"
        self.nroots = 1
        self.inverse = 1.0
        # for dmrg-hartree hybrid
        self.niterations = 20
        self.dmrg_thresh = 1e-5
        self.hartree_thresh = 1e-5