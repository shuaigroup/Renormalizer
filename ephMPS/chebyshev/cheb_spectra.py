# -*- coding: utf-8 -*-
# Author: Tong Jiang <tongjiang1000@gmail.com>

from ephMPS.mps import Mpo
import copy
import numpy as np


class ChebyshevSpectra(object):
    def __init__(
        self,
        mol_list,
        spectratype,
        freq_reg,
        dim_krylov,
        krylov_sweep
    ):
        self.mol_list = mol_list
        assert spectratype in ['abs', 'emi']
        self.spectratype = spectratype
        self.h_mpo = Mpo(mol_list)
        if spectratype == "abs":
            self.nexciton = 0
        else:
            self.nexciton = 1
        self.freq_reg = freq_reg
        self.dim_krylov = dim_krylov
        self.krylov_sweep = krylov_sweep

    def calc_termlist_one_step(self, interval=10):
        start = len(self.termlist)
        if start == 2:
            interval -= 2
        for i in range(start, start+interval):
            tmp = self.L_apply(self.t_nm1).scale(2)
            t_n = tmp.add(self.t_nm2.scale(-1))
            t_n = t_n.canonicalise()
            t_n.compress()

            print('bond dimension', [x.shape for x in t_n])
            t_n = self.truncate(t_n)

            self.t_nm2 = copy.deepcopy(self.t_nm1)
            self.t_nm1 = copy.deepcopy(t_n)

            self.termlist.append(self.firstmps.conj().dot(t_n))

    def L_apply(self, imps):
        raise NotImplementedError

    def truncate(self, t_n):
        raise NotImplementedError

    def cheb_sum(self, max_N=3000, interval=10):
        def damping_factor(tot_n):
            return [
                ((tot_n - i + 1) * np.cos(i * np.pi / (tot_n + 1)) +
                 np.sin(i * np.pi / (
                     tot_n + 1)) / np.tan(np.pi / (tot_n + 1))
                 ) / (tot_n + 1) for i in range(tot_n)
            ]
        self.init_termlist()
        max_N = 3000
        greenfunc = np.zeros((len(self.freq_reg), max_N // interval))
        cheb_freq = []
        for omega in self.freq_reg:
            cheb_freq.append([1, omega])
        # bond_dim = []
        for ith_interval in range(greenfunc.shape[1]):
            print('No.%d interval' % ith_interval)
            self.calc_termlist_one_step()
            df = damping_factor(len(self.termlist))

            freq_idx = 0
            for freq in self.freq_reg:
                if ith_interval == 0:
                    len_interval = interval - 2
                else:
                    len_interval = interval
                len_idx = len(cheb_freq[freq_idx])
                for n in range(len_idx, len_idx+len_interval):
                    cheb_freq[freq_idx].append(
                        2 * freq * cheb_freq[freq_idx][n - 1] -
                        cheb_freq[freq_idx][n - 2]
                    )
                G_list = [df[0] * self.termlist[0]] + \
                    [2 * df[i] * cheb_freq[freq_idx][i] * self.termlist[i]
                     for i in range(1, len(self.termlist))]
                greenfunc[freq_idx][ith_interval] = (1. / np.sqrt(1 - omega**2)
                                                     * sum(G_list))
                freq_idx += 1
            np.save('chebshev.npy', greenfunc[:, :ith_interval+1])
        return greenfunc
