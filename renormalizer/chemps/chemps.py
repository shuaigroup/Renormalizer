from renormalizer.mps import Mpo
from renormalizer.mps.backend import np
import logging

logger = logging.getLogger(__name__)


class CheMps(object):
    def __init__(
            self,
            model,
            freq,
            max_iter,
            batch_num
    ):
        self.model = model
        self.h_mpo = Mpo(model)
        self.freq = freq
        self.max_iter = max_iter
        self.batch_num = batch_num

    def init_moment(self):
        raise NotImplementedError

    def evolve(self):
        i_step = 2
        first_mps, t_nm2, t_nm1, moment_list = self.init_moment()
        logger.info(f"spectra will be calculated with evolution interval:{self.batch_num}")
        response = []

        def generate_che_omega(w, n):
            freq_list = [1, w]
            for i_w in range(n):
                freq_list.append(2 * w * freq_list[-1] - freq_list[-2])
            return freq_list

        while i_step <= self.max_iter:
            logger.info(f"generating No.{i_step} vector...")
            t_nm2, t_nm1 = self.generate(t_nm2, t_nm1)
            moment_list.append(t_nm1.conj().dot(first_mps))
            if i_step % self.batch_num == 0:
                damping_f = [((i_step - i + 1) * np.cos(i * np.pi / (i_step + 1)) +
                              np.sin(i * np.pi / (i_step + 1)) / np.tan(np.pi / (i_step + 1))
                              ) / (i_step + 1) for i in range(i_step)
                             ]
                logger.info(f"collecting spectral intensity")
                batch_response = []
                for omega in self.freq:
                    che_freq = generate_che_omega(omega, i_step)
                    collect = [damping_f[0] * moment_list[0]] + \
                              [2 * damping_f[i] * che_freq[i] * moment_list[i]
                               for i in range(1, i_step)]
                    batch_response.append(1. / np.sqrt(1 - omega**2) * sum(collect))
                response.append(batch_response)
            i_step = i_step + 1
        return response
