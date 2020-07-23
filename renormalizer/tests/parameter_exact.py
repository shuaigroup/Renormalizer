# -*- coding: utf-8 -*-
import numpy as np

from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.utils import Quantity
from renormalizer.utils.qutip_utils import get_clist, get_blist, get_holstein_hamiltonian, get_gs


OMEGA = 1
DISPLACEMENT = 1
N_LEVELS = 2
N_SITES = 3
J = 1

ph = Phonon.simple_phonon(Quantity(OMEGA), Quantity(DISPLACEMENT), N_LEVELS)
mol = Mol(Quantity(0), [ph])
model = HolsteinModel([mol] * N_SITES, Quantity(J), 3)

qutip_clist = get_clist(N_SITES, N_LEVELS)
qutip_blist = get_blist(N_SITES, N_LEVELS)

G = np.sqrt(DISPLACEMENT**2 * OMEGA / 2)
qutip_h = get_holstein_hamiltonian(N_SITES, J, OMEGA, G, qutip_clist, qutip_blist)

qutip_gs = get_gs(N_SITES, N_LEVELS)