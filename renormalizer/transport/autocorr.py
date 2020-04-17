# -*- coding: utf-8 -*-

import logging

import scipy.integrate

from renormalizer.mps import MpDm, Mpo, BraKetPair, ThermalProp, load_thermal_state
from renormalizer.mps.backend import np
from renormalizer.mps.lib import compressed_sum
from renormalizer.utils.constant import mobility2au
from renormalizer.utils import TdMpsJob, Quantity, EvolveConfig, CompressConfig, Op
from renormalizer.utils.utils import cast_float
from renormalizer.model import MolList, MolList2, ModelTranslator
from renormalizer.property import Property

logger = logging.getLogger(__name__)


class TransportAutoCorr(TdMpsJob):

    def __init__(self, mol_list, temperature: Quantity, j_oper: Mpo =None,
            insteps: int=1, ievolve_config=None, compress_config=None,
            evolve_config=None, dump_dir: str=None, job_name: str=None, properties: Property = None,):
        self.mol_list = mol_list
        self.h_mpo = Mpo(mol_list)
        if j_oper is None:
            self.j_oper = self._construct_flux_operator()
        else:
            self.j_oper = j_oper
        self.temperature = temperature

        # imaginary time evolution config
        if ievolve_config is None:
            self.ievolve_config = EvolveConfig()
            if insteps is None:
                self.ievolve_config.adaptive = True
                # start from a small step
                self.ievolve_config.guess_dt = temperature.to_beta() / 1e5j
                insteps = 1
        else:
            self.ievolve_config = ievolve_config
        self.insteps = insteps

        if compress_config is None:
            logger.debug("using default compress config")
            self.compress_config = CompressConfig()
        else:
            self.compress_config = compress_config

        self.impdm = None
        self.properties = properties
        self._auto_corr = []
        super().__init__(evolve_config=evolve_config, dump_dir=dump_dir,
                job_name=job_name)


    def _construct_flux_operator(self):
        # construct flux operator
        logger.info("constructing 1-d Holstein model flux operator ")
        
        if isinstance(self.mol_list, MolList):
            if self.mol_list.periodic:
                itera = range(len(self.mol_list)) 
            else:
                itera = range(len(self.mol_list)-1) 
            j_list = []
            for i in itera:
                conne = (i+1) % len(self.mol_list) # connect site index 
                j1 = Mpo.intersite(self.mol_list, {i:r"a", conne:r"a^\dagger"}, {},
                        Quantity(self.mol_list.j_matrix[i, conne]))
                j1.compress_config.threshold = 1e-8
                j2 = j1.conj_trans().scale(-1)
                j_list.extend([j1, j2])
            j_oper = compressed_sum(j_list, batchsize=10)
        
        elif isinstance(self.mol_list, MolList2):
            
            # In multi_electron case, the zero-exciton state is not guaranteed to be added as
            # scheme4, it is more flexiable depending on the definition of
            # mol_list.order and mol_list.model
            assert not self.mol_list.multi_electron
            
            e_nsite = self.mol_list.e_nsite
            model = {}
            for i in range(e_nsite):
                conne = (i+1) % e_nsite # connect site index 
                model[(f"e_{i}", f"e_{conne}")] = [(Op(r"a^\dagger",1), Op("a", -1),
                    -self.mol_list.j_matrix[i, conne]), (Op(r"a",-1), Op(r"a^\dagger", 1),
                    self.mol_list.j_matrix[conne, i])]
            j_oper = Mpo.general_mpo(self.mol_list, model=model,
                    model_translator=ModelTranslator.general_model)
        else:
            assert False
        logger.debug(f"flux operator bond dim: {j_oper.bond_dims}")
        
        return j_oper

    def init_mps(self):
        # first try to load
        if self._defined_output_path:
            mpdm = load_thermal_state(self.mol_list, self._thermal_dump_path)
        else:
            mpdm = None
        # then try to calculate
        if mpdm is None:
            i_mpdm = MpDm.max_entangled_ex(self.mol_list)
            i_mpdm.compress_config = self.compress_config
            if self.job_name is None:
                job_name = None
            else:
                job_name = self.job_name + "_thermal_prop"
            tp = ThermalProp(i_mpdm, self.h_mpo, evolve_config=self.ievolve_config, dump_dir=self.dump_dir, job_name=job_name)
            # only propagate half beta
            tp.evolve(None, self.insteps, self.temperature.to_beta() / 2j)
            mpdm = tp.latest_mps
            if self._defined_output_path:
                mpdm.dump(self._thermal_dump_path)
        self.impdm = mpdm
        self.impdm.compress_config = self.compress_config
        e = mpdm.expectation(self.h_mpo)
        self.h_mpo = Mpo(self.mol_list, offset=Quantity(e))
        mpdm.evolve_config = self.evolve_config
        ket_mpdm = self.j_oper.contract(mpdm).canonical_normalize()
        bra_mpdm = mpdm.copy()
        return BraKetPair(bra_mpdm, ket_mpdm, self.j_oper)

    def process_mps(self, mps):
        self._auto_corr.append(mps.ft)

        # calculate other properties defined in Property
        if self.properties is not None:
            self.properties.calc_properties_braketpair(mps)

    def evolve_single_step(self, evolve_dt):
        prev_bra_mpdm, prev_ket_mpdm = self.latest_mps
        latest_ket_mpdm = prev_ket_mpdm.evolve(self.h_mpo, evolve_dt)
        latest_bra_mpdm = prev_bra_mpdm.evolve(self.h_mpo, evolve_dt)
        return BraKetPair(latest_bra_mpdm, latest_ket_mpdm, self.j_oper)

    def stop_evolve_criteria(self):
        corr = self.auto_corr
        if len(corr) < 10:
            return False
        last_corr = corr[-10:]
        first_corr = corr[0]
        return np.abs(last_corr.mean()) < 1e-5 * np.abs(first_corr) and last_corr.std() < 1e-5 * np.abs(first_corr)

    @property
    def auto_corr(self):
        return np.array(self._auto_corr)

    def get_dump_dict(self):
        dump_dict = dict()
        dump_dict["mol list"] = self.mol_list.to_dict()
        dump_dict["temperature"] = self.temperature.as_au()
        dump_dict["time series"] = self.evolve_times
        dump_dict["auto correlation real"] = cast_float(self.auto_corr.real)
        dump_dict["auto correlation imag"] = cast_float(self.auto_corr.imag)
        dump_dict["mobility"] = self.calc_mobility()[1]
        if self.properties is not None:
            for prop_str in self.properties.prop_res.keys():
                dump_dict[prop_str] = self.properties.prop_res[prop_str]
        
        return dump_dict

    def calc_mobility(self):
        time_series = self.evolve_times
        corr_real = self.auto_corr.real
        inte = scipy.integrate.trapz(corr_real, time_series)
        mobility_in_au = inte / self.temperature.as_au()
        mobility = mobility_in_au / mobility2au
        return mobility_in_au, mobility
