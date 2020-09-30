from renormalizer.chemps.chemps import CheMps
from renormalizer.mps.backend import np, USE_GPU, xp
from renormalizer.mps import Mpo, Mps, gs, MpDm, ThermalProp, load_thermal_state
from renormalizer.utils import OptimizeConfig, CompressConfig, CompressCriteria, EvolveConfig
from renormalizer.mps.matrix import moveaxis, asnumpy
import opt_einsum as oe
import copy
import logging
import os

logger = logging.getLogger(__name__)


class CheMpsFiniteT(CheMps):
    def __init__(
            self,
            model,
            temperature,
            spectra_type,
            freq,
            m_max,
            full_many_body=False,
            sampling_num=1000,
            ns=10,
            dk=10,
            procedure_gs=None,
            max_iter=1000,
            batch_num=10,
            icompress_config=None,
            ievolve_config=None,
            insteps=None,
            dump_dir: str=None,
            job_name=None,
    ):
        self.spectra_type = spectra_type
        self.temperature = temperature
        self.m_max = m_max
        self.full_many_body = full_many_body
        self.sampling_num = sampling_num
        self.ns = ns
        self.dk = dk
        self.procedure_gs = procedure_gs
        if self.procedure_gs is None:
            self.procedure_gs = \
                [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]
        self.evolve_config = ievolve_config
        self.compress_config = icompress_config
        if self.evolve_config is None:
            self.evolve_config = \
                EvolveConfig()
        if self.compress_config is None:
            self.compress_config = \
                CompressConfig(CompressCriteria.fixed,
                               max_bonddim=m_max)
            self.compress_config.set_bonddim(len(model.pbond_list))
        self.insteps = insteps
        self.dump_dir = dump_dir
        self.job_name = job_name
        self.h_mpo1 = None
        self.h_mpo2 = None
        super(CheMpsFiniteT, self).__init__(model, freq, max_iter, batch_num)

    def init_mp(self):
        assert self.spectra_type in ["abs", "emi"]
        dipole_type = r"a^\dagger" if self.spectra_type == "abs" else "a"
        beta = self.temperature.to_beta()
        if self.spectra_type == "abs":
            i_mpo = MpDm.max_entangled_gs(self.model)
            tp = ThermalProp(i_mpo, self.h_mpo, exact=True, space='GS')
            tp.evolve(None, 1, beta / 2j)
            ket_mpo = tp.latest_mps

        elif self.spectra_type == "emi":
            if self._defined_output_path:
                ket_mpo = \
                    load_thermal_state(self.model, self._thermal_dump_path)
            else:
                impo = MpDm.max_entangled_ex(self.model)
                impo.compress_config = self.compress_config
                if self.job_name is None:
                    job_name = None
                else:
                    job_name = self.job_name + "_thermal_prop"
                tp = ThermalProp(
                    impo, self.h_mpo, evolve_config=self.evolve_config,
                    dump_dir=self.dump_dir, job_name=job_name)
                tp.evolve(None, self.insteps, beta / 2j)
                ket_mpo = tp.latest_mps
                if self._defined_output_path:
                    ket_mpo.dump(self._thermal_dump_path)
        else:
            assert False

        dipole_mpo = \
            Mpo.onsite(
                self.model, dipole_type, dipole=True
            )
        a_ket_mpo = dipole_mpo.apply(ket_mpo, canonicalise=True)
        # a_ket_mpo.canonical_normalize()

        def find_many_body_band():
            gs_mps_low = Mps.random(
                self.model, 0, self.procedure_gs[0][0], percent=1.0)
            gs_mps_low.optimize_config = OptimizeConfig(procedure=self.procedure_gs)
            gs_mps_low.optimize_config.method = "2site"
            gs_energies_low, gs_mps_low = gs.optimize_mps(gs_mps_low, self.h_mpo)

            gs_mps_high = Mps.random(
                self.model, 0, self.procedure_gs[0][0], percent=1.0)
            gs_mps_high.optimize_config = OptimizeConfig(procedure=self.procedure_gs)
            gs_mps_high.optimize_config.inverse = -1.0
            gs_mps_high.optimize_config.method = "2site"
            gs_energies_high, gs_mps_high = gs.optimize_mps(gs_mps_high, self.h_mpo)

            ex_mps_low = Mps.random(
                self.model, 1, self.procedure_gs[0][0], percent=1.0)
            ex_mps_low.optimize_config = OptimizeConfig(procedure=self.procedure_gs)
            ex_mps_low.optimize_config.method = "2site"
            ex_energies_low, ex_mps_low = gs.optimize_mps(ex_mps_low, self.h_mpo)

            ex_mps_high = Mps.random(
                self.model, 1, self.procedure_gs[0][0], percent=1.0)
            ex_mps_high.optimize_config = OptimizeConfig(procedure=self.procedure_gs)
            ex_mps_high.optimize_config.inverse = -1.0
            ex_mps_high.optimize_config.method = "2site"
            ex_energies_high, ex_mps_high = gs.optimize_mps(ex_mps_high, self.h_mpo)

            return ex_energies_low[-1]+gs_energies_high[-1], -ex_energies_high[-1]-gs_energies_low[-1]

        lowest_gap, highest_gap = find_many_body_band()
        logger.info(f"full many body band:{lowest_gap, highest_gap}")
        if self.full_many_body:
            if self.spectra_type is "abs":
                self.freq = np.linspace(lowest_gap, highest_gap, num=self.sampling_num)
            else:
                self.freq = np.linspace(-highest_gap, -lowest_gap, num=self.sampling_num)
        a_ket_mpo.model = self.model
        logger.info(f"e_occupation of upper bond for initial mpo:{sum(a_ket_mpo.e_occupations)}")
        logger.info(f"e_occupation of lower bond for initial mpo:{sum(a_ket_mpo.conj_trans().e_occupations)}")
        return a_ket_mpo

    def projection(self, eps=0.025):
        w_prime = 1 - 0.5 * eps
        freq_width = self.freq[-1] - self.freq[0]
        scale_factor = freq_width / (2 * w_prime)
        self.h_mpo2 = self.h_mpo.scale(1 / scale_factor)
        identity = Mpo.identity(self.model).scale(-self.freq[0]/scale_factor-w_prime)
        self.h_mpo1 = self.h_mpo2.add(identity)
        self.freq = [(ifreq - self.freq[0]) / scale_factor - w_prime for ifreq in self.freq]

    def init_moment(self):
        first_mpo = self.init_mp()
        t_nm2 = copy.deepcopy(first_mpo)
        self.projection()
        t_nm1 = self.h_mpo1.apply(t_nm2)
        t_nm1 = t_nm1.add(t_nm2.apply(self.h_mpo2).scale(-1))
        moment_list = [t_nm2.conj().dot(first_mpo),
                       t_nm1.conj().dot(first_mpo), ]
        return first_mpo, t_nm2, t_nm1, moment_list

    def generate(self, t_nm2, t_nm1):
        t_n = self.h_mpo1.apply(t_nm1)
        t_n = t_n.add(t_nm1.apply(self.h_mpo2).scale(-1))
        t_n = t_n.scale(2)
        t_n = t_n.add(t_nm2.scale(-1))
        t_n = t_n.canonicalise()
        t_n.compress_config = \
            CompressConfig(criteria=CompressCriteria.fixed,
                           max_bonddim=self.m_max)
        # logger.info(f"e_occupation of upper bond before compression:{sum(t_n.e_occupations)/(t_n.dot(t_n))}")
        t_n.compress()
        if self.spectra_type is "abs":
            t_n = t_n.apply(MpDm.max_entangled_gs(self.model, normalize=False))
        elif self.spectra_type is "emi":
            t_n = t_n.apply(MpDm.max_entangled_ex(self.model, normalize=False))
            t_n.compress()
        else:
            assert False
        # t_n.model = self.model
        # logger.info(f"e_occupation of upper bond:{sum(t_n.e_occupations)/(t_n.dot(t_n))}")
        # logger.info(f"e_occupation of lower bond:{sum(t_n.conj_trans().e_occupations)/(t_n.dot(t_n))}")

        # t_n.variational_compress()
        if not self.full_many_body:
            t_n = self.truncate(t_n, self.ns, self.dk)
        return t_nm1, t_n

    def truncate(self, t_n, ns, dk, thresh=1.0, ortho_method='lanczos'):
        def norm_local_mpo(local_mpo):
            norm2 = oe.contract("abcd, abcd", local_mpo, local_mpo)
            return np.sqrt(norm2)

        if USE_GPU:
            oe_backend = "cupy"
        else:
            oe_backend = "numpy"

        lr1 = [np.ones((1, 1, 1))]
        lr2 = [np.ones((1, 1, 1))]

        for i_site in range(1, len(t_n)):
            i_site_dagger = moveaxis(t_n[i_site-1], (1, 2), (2, 1))
            lr1.append(
                oe.contract("abc, adef, begh, cgdi->fhi",
                            lr1[-1], i_site_dagger, self.h_mpo1[i_site-1],
                            t_n[i_site-1], backend=oe_backend
                            )
            )
            lr2.append(
                oe.contract("abc, adef, begh, cgdi->fhi",
                            lr2[-1], t_n[i_site-1], self.h_mpo2[i_site-1],
                            i_site_dagger, backend=oe_backend
                            )
            )
        lr1.append(np.ones((1, 1, 1)))
        lr2.append(np.ones((1, 1, 1)))

        i_sweep = 0
        while i_sweep < ns:
            if i_sweep % 2 == 0:
                sweep_order = range(len(self.h_mpo), 1, -1)
            else:
                sweep_order = range(1, len(self.h_mpo))
            for i_site in sweep_order:
                # logger.info(f"begin truncate at site:{i_site}")
                reno_h_1 = oe.contract("abc, bdef, gfh->adgceh",
                                       lr1[i_site-1], self.h_mpo1[i_site-1], lr1[i_site]
                                       )

                reno_h_2 = oe.contract("abc, bdef, gfh->adgceh",
                                       lr2[i_site-1], self.h_mpo2[i_site-1], lr2[i_site]
                                       )

                if ortho_method is 'lanczos':
                    ortho_vectors = [np.array(t_n[i_site-1]) / norm_local_mpo(np.array(t_n[i_site-1]))]
                    beta = []
                    alpha = []
                    for j_v in range(1, dk+1):
                        if j_v == 1:
                            wj = oe.contract("abcdef, degf->abgc", reno_h_1, ortho_vectors[-1]) - \
                                 oe.contract("agbc, abcdef->dgef", ortho_vectors[-1], reno_h_2)
                        else:
                            wj = oe.contract("abcdef, degf->abgc", reno_h_1, ortho_vectors[-1]) - \
                                 oe.contract("agbc, abcdef->dgef", ortho_vectors[-1], reno_h_2) - \
                                 beta[-1] * ortho_vectors[-2]
                        alpha.append(oe.contract("abcd, abcd", wj, ortho_vectors[-1]))
                        wj = wj - alpha[-1] * ortho_vectors[-1]
                        beta.append(norm_local_mpo(wj))
                        if beta[-1] <= 1.e-3:
                            break
                        else:
                            ortho_vectors.append(wj / beta[-1])
                else:
                    raise NotImplementedError
                krylov_h = np.diag(alpha) + np.diag(beta[:-1], k=-1) + \
                    np.diag(beta[:-1], k=1)
                eigen_w, eigen_v = np.linalg.eigh(krylov_h)
                throw_away = np.where(eigen_w >= thresh)
                if len(throw_away[0])>0:
                    logger.info(f"vector with energy beyond 1:{throw_away}")
                update_t_n = np.array(t_n[i_site-1])
                for idx in throw_away[0]:
                    proj_op = eigen_v[0, idx] * ortho_vectors[0]
                    for jbasis in range(1, len(alpha)):
                        proj_op = proj_op + eigen_v[jbasis, idx] * ortho_vectors[jbasis]
                    update_t_n = update_t_n - oe.contract("abcd, abcd", proj_op, t_n[i_site-1]) * proj_op
                overlap = oe.contract("abcd, abcd", t_n[i_site-1], update_t_n)
                # logger.info(
                #     f"truncation-induced change:{norm_local_mpo(update_t_n)**2+norm_local_mpo(t_n[i_site-1])**2-2*overlap}")
                t_n[i_site-1] = update_t_n
                i_site_dagger = moveaxis(t_n[i_site-1], (1, 2), (2, 1))
                if i_sweep % 2 == 0:
                    lr1[i_site-1] = oe.contract("abc, defa, gfhb, ihec->dgi",
                                                lr1[i_site], i_site_dagger, self.h_mpo1[i_site-1],
                                                t_n[i_site-1], backend=oe_backend
                                                )
                    lr2[i_site-1] = oe.contract("abc, defa, gfhb, ihec->dgi",
                                                lr2[i_site], t_n[i_site-1], self.h_mpo2[i_site-1],
                                                i_site_dagger, backend=oe_backend
                                                )
                else:
                    lr1[i_site] = oe.contract("abc, adef, begh, cgdi->fhi",
                                              lr1[i_site-1], i_site_dagger, self.h_mpo1[i_site-1],
                                              t_n[i_site-1], backend=oe_backend
                                              )
                    lr2[i_site] = oe.contract("abc, adef, begh, cgdi->fhi",
                                              lr2[i_site-1], t_n[i_site-1], self.h_mpo2[i_site-1],
                                              i_site_dagger, backend=oe_backend
                                              )
            i_sweep = i_sweep + 1
        return t_n

    @property
    def _thermal_dump_path(self):
        assert self._defined_output_path
        return os.path.join(self.dump_dir, self.job_name + "_impo.npz")

    @property
    def _defined_output_path(self):
        return self.dump_dir is not None and self.job_name is not None















