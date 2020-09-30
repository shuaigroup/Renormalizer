from renormalizer.chemps.chemps import CheMps
from renormalizer.mps.backend import np, USE_GPU
from renormalizer.mps import Mpo, Mps, gs
from renormalizer.utils import OptimizeConfig, CompressConfig, CompressCriteria
import opt_einsum as oe
import copy
import logging

logger = logging.getLogger(__name__)


class CheMpsZeroT(CheMps):
    def __init__(
            self,
            model,
            spectra_type,
            freq,
            m_max,
            full_many_body=False,
            sampling_num=1000,
            ns=10,
            dk=10,
            procedure_gs=None,
            max_iter=1000,
            batch_num=10
    ):
        self.spectra_type = spectra_type
        self.m_max = m_max
        self.full_many_body = full_many_body
        self.sampling_num = sampling_num
        self.ns = ns
        self.dk = dk
        self.procedure_gs = procedure_gs
        if self.procedure_gs is None:
            self.procedure_gs = \
                [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]
        super(CheMpsZeroT, self).__init__(model, freq, max_iter, batch_num)

    def init_mp(self):
        assert self.spectra_type in ["abs", "emi"]
        n_exciton = 0 if self.spectra_type == "abs" else 1
        dipole_type = r"a^\dagger" if self.spectra_type == "abs" else "a"
        mps = Mps.random(
            self.model, n_exciton, self.procedure_gs[0][0], percent=1.0)
        mps.optimize_config = OptimizeConfig(procedure=self.procedure_gs)
        mps.optimize_config.method = "2site"
        energies, mps = gs.optimize_mps(mps, self.h_mpo)
        dipole_mpo = \
            Mpo.onsite(
                self.model, dipole_type, dipole=True
            )
        a_ket_mps = dipole_mpo.apply(mps, canonicalise=True)
        # a_ket_mps.canonical_normalize()

        def find_many_body_band():
            mps_low = Mps.random(
                self.model, 1, self.procedure_gs[0][0], percent=1.0)
            mps_low.optimize_config = OptimizeConfig(procedure=self.procedure_gs)
            mps_low.optimize_config.method = "2site"
            energies_low, mps_low = gs.optimize_mps(mps_low, self.h_mpo)

            mps_high = Mps.random(
                self.model, 1, self.procedure_gs[0][0], percent=1.0)
            mps_high.optimize_config.inverse = -1.0
            energies_high, mps_high = gs.optimize_mps(mps_high, self.h_mpo)

            return energies_low[-1], -energies_high[-1]

        if self.full_many_body:
            lowest_e, highest_e = find_many_body_band()
            lowest_e = lowest_e - energies[-1]
            highest_e = highest_e - energies[-1]
            logger.info(f"the full many body band is{lowest_e, highest_e}")
            width = highest_e - lowest_e
            lowest_e = lowest_e - width / 10
            self.freq = np.linspace(lowest_e, highest_e, num=self.sampling_num)

        return energies[-1], a_ket_mps

    def projection(self, e0, eps=0.025):
        w_prime = 1 - 0.5 * eps
        freq_width = self.freq[-1] - self.freq[0]
        scale_factor = freq_width / (2 * w_prime)
        identity = Mpo.identity(self.model).scale(-self.freq[0]-e0)
        self.h_mpo = self.h_mpo.add(identity)
        self.h_mpo = self.h_mpo.scale(1 / scale_factor)
        identity = Mpo.identity(self.model).scale(-w_prime)
        self.h_mpo = self.h_mpo.add(identity)

        self.freq = [(ifreq - self.freq[0]) / scale_factor - w_prime for ifreq in self.freq]

    def init_moment(self):
        e0, first_mps = self.init_mp()
        t_nm2 = copy.deepcopy(first_mps)
        self.projection(e0)
        t_nm1 = self.h_mpo.apply(t_nm2)
        moment_list = [t_nm2.conj().dot(first_mps),
                       t_nm1.conj().dot(first_mps), ]
        return first_mps, t_nm2, t_nm1, moment_list

    def generate(self, t_nm2, t_nm1):
        t_n = self.h_mpo.apply(t_nm1).scale(2)
        t_n = t_n.add(t_nm2.scale(-1))
        t_n = t_n.canonicalise()
        t_n.compress_config = \
            CompressConfig(criteria=CompressCriteria.fixed,
                           max_bonddim=self.m_max)
        t_n.compress()
        # t_n.variational_compress()
        if not self.full_many_body:
            t_n = self.truncate(t_n, self.ns, self.dk)
        return t_nm1, t_n

    def truncate(self, t_n, ns, dk, thresh=1.0, ortho_method='lanczos'):
        def norm_local_mps(local_mps):
            norm2 = oe.contract("abc, abc", local_mps, local_mps)
            return np.sqrt(norm2)

        if USE_GPU:
            oe_backend = "cupy"
        else:
            oe_backend = "numpy"
        # oe_backend = 'numpy'

        lr = [np.ones((1, 1, 1))]
        for i_site in range(1, len(t_n)):
            lr.append(
                oe.contract("abc, ade, bdfg, cfh->egh",
                            lr[-1], t_n[i_site-1], self.h_mpo[i_site-1],
                            t_n[i_site-1], backend=oe_backend
                            )
            )
        lr.append(np.ones((1, 1, 1)))
        i_sweep = 0
        while i_sweep < ns:
            if i_sweep % 2 == 0:
                sweep_order = range(len(self.h_mpo), 1, -1)
            else:
                sweep_order = range(1, len(self.h_mpo))
            for i_site in sweep_order:
                reno_h = oe.contract("abc, bdef, gfh->adgceh",
                                     lr[i_site-1], self.h_mpo[i_site-1], lr[i_site]
                                     )
                if ortho_method is 'lanczos':
                    ortho_vectors = [np.array(t_n[i_site-1]) / norm_local_mps(np.array(t_n[i_site-1]))]
                    beta = []
                    alpha = []
                    for j_v in range(1, dk+1):
                        if j_v == 1:
                            wj = oe.contract("abcdef, def->abc", reno_h, ortho_vectors[-1])
                        else:
                            wj = oe.contract("abcdef, def->abc", reno_h, ortho_vectors[-1]) - beta[-1] * ortho_vectors[-2]
                        alpha.append(oe.contract("abc, abc", wj, ortho_vectors[-1]))
                        wj = wj - alpha[-1] * ortho_vectors[-1]
                        beta.append(norm_local_mps(wj))
                        if beta[-1] <= 1.e-3:
                            break
                        else:
                            ortho_vectors.append(wj / beta[-1])
                krylov_h = np.diag(alpha) + np.diag(beta[:-1], k=-1) + \
                    np.diag(beta[:-1], k=1)
                eigen_w, eigen_v = np.linalg.eigh(krylov_h)
                throw_away = np.where(np.abs(eigen_w) >= thresh)
                update_t_n = np.array(t_n[i_site-1])
                for idx in throw_away[0]:
                    proj_op = eigen_v[0, idx] * ortho_vectors[0]
                    for jbasis in range(1, len(alpha)):
                        proj_op = proj_op + eigen_v[jbasis, idx] * ortho_vectors[jbasis]
                    update_t_n = update_t_n - oe.contract("abc, abc", proj_op, t_n[i_site-1]) * proj_op
                overlap = oe.contract("abc, abc", t_n[i_site-1], update_t_n)
                if len(throw_away[0])>0:
                    logger.info(f"vector with energy beyond 1:{throw_away}")
                    logger.info(
                    f"truncation-induced change:{norm_local_mps(update_t_n)**2+norm_local_mps(t_n[i_site-1])**2-2*overlap}")
                t_n[i_site-1] = update_t_n
                if i_sweep % 2 == 0:
                    lr[i_site-1] = oe.contract("abc, dea, fegb, hgc->dfh",
                                               lr[i_site], t_n[i_site-1], self.h_mpo[i_site-1],
                                               t_n[i_site-1], backend=oe_backend
                                               )
                else:
                    lr[i_site] = oe.contract("abc, ade, bdfg, cfh->egh",
                                             lr[i_site-1], t_n[i_site-1], self.h_mpo[i_site-1],
                                             t_n[i_site-1], backend=oe_backend
                                             )
            i_sweep = i_sweep + 1
        return t_n















