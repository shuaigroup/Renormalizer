# -*- coding: utf-8 -*-
import numpy as np

from renormalizer.mps import Mpo
from renormalizer.model import Phonon, Mol, MolList
from renormalizer.utils import Quantity
from renormalizer.utils.qutip_utils import get_clist, get_blist, get_hamiltonian, get_gs


OMEGA = 1
DISPLACEMENT = 1
N_LEVELS = 2
N_SITES = 3
J = 1

ph = Phonon.simple_phonon(Quantity(OMEGA), Quantity(DISPLACEMENT), N_LEVELS)
mol = Mol(Quantity(0), [ph])
mol_list = MolList([mol] * N_SITES, Quantity(J), 3)

qutip_clist = get_clist(N_SITES, N_LEVELS)
qutip_blist = get_blist(N_SITES, N_LEVELS)

G = np.sqrt(DISPLACEMENT**2 * OMEGA / 2)
qutip_h = get_hamiltonian(N_SITES, J, OMEGA, G, qutip_clist, qutip_blist)

qutip_gs = get_gs(N_SITES, N_LEVELS)