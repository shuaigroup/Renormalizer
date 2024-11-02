# -*- coding: utf-8 -*-
'''
only works for multi_electron state
'''

from renormalizer.utils import TdMpsJob
from renormalizer.utils.constant import *
from renormalizer.mps.backend import xp, np, OE_BACKEND
from renormalizer.mps.matrix import asxp, asnumpy
from renormalizer.mps import Mpo, Mps, MpDm, gs, ThermalProp
from renormalizer.mps.mps import BraKetPair
from renormalizer.mps.lieq import solve_mps
from renormalizer.utils import OptimizeConfig, EvolveConfig, CompressConfig, constant, CompressCriteria
from renormalizer.model import Op, Model
#from renormalizer.transport.kubo import current_op
from renormalizer.model import basis as ba
from renormalizer.mps.lib import Environ

import logging
import itertools
import os
import opt_einsum as oe
import scipy.linalg

logger = logging.getLogger(__name__)


def single_mol_model(fdusin, fnac, projector=0):
    w0 = []
    d0 = []      # project on PES0 normal coordinates
    w1 = []
    d1 = []      # project on PES1 normal coordinates 
    s021 = [] 
    s120 = []
    with open(fdusin, "r") as f:
        lines = f.readlines()
        read_start = False
        for line in lines:
            if line[:6] == "------": 
                if read_start:
                    break
                else:
                    read_start = True
                    continue
            if read_start:
                split_line = line.split()
                w0.append(float(split_line[3]))
                d0.append(float(split_line[4]))
                w1.append(float(split_line[9]))       
                d1.append(float(split_line[10]))       
        
        nmodes = len(w0)

        start_s120 = start_s021 = False
        for iline, line in enumerate(lines):
            if line.rstrip().lstrip() == "BEGIN_DUSH_DATA_1":
                start_s021 = True   # s021: S matrix (x_0, x_1)
                start_s120 = False  # s120: S matrix (x_1, x_0)
            if line.rstrip().lstrip() == "BEGIN_DUSH_DATA_2":
                start_s120 = True
                start_s021 = False
            if start_s120 or start_s021:
                if line.split()[0] == "MODE":
                    res = [] 
                    for subline in lines[iline+1:iline+int(np.ceil(nmodes/10))+1]:
                        res.extend([float(i) for i in subline.split()])
                    if start_s120:
                        s120.append(res)   
                    elif start_s021:
                        s021.append(res)   
    
    nac = []
    if fnac is not None:
        with open(fnac, "r") as f:
            lines = f.readlines()
            for iline, line in enumerate(lines):
                split_line = line.split()
                if len(split_line) > 0:
                    if split_line[0] == "m" and split_line[1] == "n":
                        for subline in lines[iline+2:iline+2+nmodes]:
                            nac.append(float(subline.split()[4]))
    
    s021 = np.stack(s021, axis=0)               
    s120 = np.stack(s120, axis=0)
    assert np.allclose(s021.dot(s021.T), np.eye(nmodes), atol=1e-6)
    assert np.allclose(s120.dot(s120.T), np.eye(nmodes), atol=1e-6)
    assert np.allclose(s120, s021.T)
    nmodes -= projector
    w0 = np.array(w0[projector:]) * constant.cm2au
    w1 = np.array(w1[projector:]) * constant.cm2au
    d0 = np.array(d0[projector:])
    d1 = np.array(d1[projector:])
    nac = np.array(nac[projector:])
    
    s021 = s021[projector:, projector:]
    s120 = s120[projector:, projector:]
    
    return w0, w1, d0, d1, nac, s021, s120
    

def abs_dipole_op(model):
    """
    excitation between 0 -> 1
    dipole contains dipole transition integral
    """
    dipole_terms = []
    assert "dipole" in model.para.keys()
    for key, value in model.para["dipole"].items():
        siteidx = model.dof_to_siteidx[key[0]]
        local_basis = model.basis[siteidx]
        idx0 = local_basis.dof.index(key[0])
        idx1 = local_basis.dof.index(key[1])
        if local_basis.sigmaqn[idx0] == 1 and local_basis.sigmaqn[idx1] == 0:
            dof = list(key)
        else:
            dof = list(key[::-1])
        print("dof", dof, value)
        dipole_terms.append(Op("a^\dagger a", dof, value, qn=[1,0]))
    
    return Mpo(model, terms=dipole_terms)

def emi_dipole_op(model):
    # the constant coupling operator (condon approximation)
    return abs_dipole_op(model).conj_trans()

def nac_op(model):
    # the momentum operator coupling
    # the quantum number is gs: 0 ex:1
    if "nac_mpo" in model.mpos.keys():
        logger.info("load nac_mpo form model.mpos")
        return model.mpos["nac_mpo"]
    else:
        nac_terms = []
        for key, value in model.para["nac"].items():
            siteidx = model.dof_to_siteidx[key[0]]
            local_basis = model.basis[siteidx]
            idx0 = local_basis.dof.index(key[0])
            idx1 = local_basis.dof.index(key[1])
            assert local_basis.sigmaqn[idx0] == 0 and local_basis.sigmaqn[idx1] == 1
            nac_terms.append(Op("a^\dagger a partialx", list(key),
                factor=-value, qn=[0,-1,0]))       
        return Mpo(model, terms=nac_terms)

def spin_op(model):
    # spin operator
    spin_terms = []
    for key, value in model.para["spin"].items():
        spin_terms.append(Op(value, key, factor=1, qn=0))       
    return Mpo(model, terms=spin_terms)

class CorrFuncBase(TdMpsJob):
    r'''Abstract class 
    <A^\dagger B>
    '''
    def __init__(
            self,
            model,
            imps_qntot,
            temperature,
            imps = None,
            optimize_config=None,
            compress_config=None,
            evolve_config=None,
            dump_mps=None,
            dump_dir=None,
            job_name=None,
        ):

        self.model = model
        if "h_mpo" in model.mpos.keys():
            logger.info("load h_mpo form model.mpos")
            self.h_mpo = model.mpos["h_mpo"]
        else:
            self.h_mpo = Mpo(model)
        self.temperature = temperature
        self.imps = imps
        self.compress_config = compress_config
        if self.compress_config is None:
            self.compress_config = CompressConfig()
        self.optimize_config = optimize_config
        if self.optimize_config is None:
            self.optimize_config = OptimizeConfig()
        self.imps_qntot = imps_qntot
        self.dump_dir = dump_dir
        self.job_name = job_name
        self.imps = self.init_imps()
        self.op_b = self.init_op_b()
        self.op_a = self.init_op_a()

        self.e_imps = self.imps.expectation(self.h_mpo)

        self._autocorr = []
        self._autocorr_time = []

        self.e_ket = self.e_bra = 0

        super(CorrFuncBase, self).__init__(evolve_config=evolve_config,
                dump_mps=dump_mps, dump_dir=dump_dir, job_name=job_name)
        
        self.latest_mps.ket_mps.evolve_config = self.latest_mps.bra_mps.evolve_config = self.evolve_config
        self.latest_mps.ket_mps.compress_config = self.latest_mps.bra_mps.compress_config = self.compress_config
        
        self.e_ket = self.latest_mps.ket_mps.expectation(self.h_mpo)
        self.e_bra = self.latest_mps.bra_mps.expectation(self.h_mpo)
        logger.debug(f"e_ket:{self.e_ket}, e_bra:{self.e_bra}")
        self.ket_mpo = self.h_mpo - self.e_ket
        self.bra_mpo = self.h_mpo - self.e_bra
        
    def pruner(self, braket_pair):
        bra_mps, ket_mps = braket_pair
        bra_mps.compress_config = ket_mps.compress_config = self.compress_config
        if self.evolve_config.is_tdvp:
            logger.debug("expand ket mps.")
            ket_mps = ket_mps.expand_bond_dimension(self.h_mpo,
                    include_ex=False)
            logger.debug("expand bra mps.")
            bra_mps = bra_mps.expand_bond_dimension(self.h_mpo,
                    include_ex=False)
        logger.debug(f"compress ket mps.")
        ket_mps.canonicalise().compress().normalize("mps_only")
        logger.debug("compress bra mps.")
        bra_mps.canonicalise().compress().normalize("mps_only")

        return BraKetPair(bra_mps, ket_mps, braket_pair.mpo)


    def init_op_a(self):
        raise NotImplementedError
    
    def init_op_b(self):
        raise NotImplementedError

    def init_imps(self):
        raise NotImplementedError

    @property
    def autocorr(self):
        return np.array(self._autocorr)
    
    @property
    def autocorr_time(self):
        return np.array(self._autocorr_time)
    

    def get_dump_dict(self):
        dump_dict = dict()
        dump_dict["correlation function time"] = self.autocorr_time
        dump_dict["ACF"] = self.autocorr
        dump_dict["time series"] = self.evolve_times
        #if self.properties is not None:
        #    for prop_str in self.properties.prop_res.keys():
        #        dump_dict[prop_str] = self.properties.prop_res[prop_str]
        
        return dump_dict



class FTCorrFuncBase(CorrFuncBase):
    r'''Abstract class 
    '''
    
    def __init__(
            self,
            model,
            imps_qntot,
            temperature,
            insteps,
            imps = None,
            ievolve_config=None,
            icompress_config=None,
            compress_config=None,
            evolve_config=None,
            dump_mps=None,
            dump_dir=None,
            job_name=None,
            optimize_config=None,
        ):
    
        self.insteps = insteps
        self.ievolve_config = ievolve_config
        self.icompress_config = icompress_config
        if self.icompress_config is None:
            self.icompress_config = CompressConfig()
        if self.ievolve_config is None:
            self.ievolve_config = EvolveConfig()
        
        super(FTCorrFuncBase, self).__init__(
                model,
                imps_qntot,
                temperature,
                imps=imps,
                compress_config=compress_config,
                evolve_config=evolve_config,
                dump_mps=dump_mps,
                dump_dir=dump_dir,
                optimize_config=optimize_config,
                job_name=job_name,
                )
    

    def init_mps(self):
        ket_mps = self.op_b.apply(self.imps)
        ket_mps.normalize("mps_norm_to_coeff")
        bra_mps = self.imps
            
        return self.pruner(BraKetPair(bra_mps, ket_mps, mpo=self.op_a.conj_trans()))
    
    def init_imps(self):
        if self.imps is not None:
            return self.imps

        mpdm = MpDm.max_entangled_mpdm(self.model, self.imps_qntot)
        mpdm.compress_config = self.icompress_config
        tp = ThermalProp(
            mpdm, self.h_mpo, evolve_config=self.ievolve_config,
            dump_dir=self.dump_dir, job_name=self.job_name, include_ex=False
        )
        if tp._defined_output_path:
            try:
                logger.info(
                    f"load density matrix from {self._thermal_dump_path}"
                )
                mpdm = MpDm.load(self.model, self._thermal_dump_path)
                logger.info(f"density matrix loaded:{mpdm}")
                return mpdm
            except FileNotFoundError:
                logger.debug(f"no file found in {self._thermal_dump_path}")
                logger.info(f"calculate mpdm from scratch.")
        
        tp.evolve(None, self.insteps, self.temperature.to_beta() / 2j)
        mpdm = tp.latest_mps
        if tp._defined_output_path:
            mpdm.dump(self._thermal_dump_path)
        
        return mpdm

    def evolve_single_step(self, evolve_dt):
        bra_mps, ket_mps = self.latest_mps
        bra_mps = bra_mps.evolve(self.bra_mpo, evolve_dt)
        ket_mps = ket_mps.evolve(self.ket_mpo, evolve_dt)
        return BraKetPair(bra_mps, ket_mps, mpo=self.latest_mps.mpo)

    def process_mps(self, braket_pair):
        t = self.evolve_times[-1]
        self._autocorr.append(braket_pair.ft * np.exp(1.0j*t*(self.e_bra - self.e_ket)))
        self._autocorr_time.append(t)
        
        #bra_mps, ket_mps = braket_pair
        #bra_rdm = bra_mps.calc_reduced_density_matrix()
        #ket_rdm = ket_mps.calc_reduced_density_matrix()
        #self.e_rdm.append([bra_rdm, ket_rdm])
    
    @property
    def _thermal_dump_path(self):
        assert self._defined_output_path
        return os.path.join(self.dump_dir, self.job_name + "_impo.npz")

class FTCorrFuncHolsteinStaticDisorder(FTCorrFuncBase):
    def __init__(
            self,
            model,
            imps_qntot,
            temperature,
            insteps,
            nmols,
            w_intra,
            w_sd,
            sd_algo,
            nsamples = None,
            optimize_config = None,
            imps = None,
            ievolve_config=None,
            icompress_config=None,
            evolve_config=None,
            compress_config=None,
            dump_mps=None,
            dump_dir=None,
            job_name=None,
        ):

        self.sd_algo = sd_algo
        self.w_sd = w_sd
        self.w_intra = w_intra
        self.nmols = nmols
        # check the sd site is on the left hand side of the chain
        sd_idx = []
        for ibas, bas in enumerate(model.basis):
            if isinstance(bas, ba.BasisSHODVR) or isinstance(bas, ba.BasisSineDVR):
                sd_idx.append(ibas)
        self.nsites_sd = len(sd_idx)
        assert np.allclose(sd_idx, np.arange(self.nsites_sd))
        
        
        if isinstance(model.basis[0], ba.BasisSHODVR):
            self.x = model.basis[0].dvr_x
            self.v = model.basis[0].dvr_v[0,:]
        elif isinstance(model.basis[0], ba.BasisSineDVR):
            self.x = model.basis[0].dvr_x
            self.v = self.wfn_sd(self.x)
            self.v /= scipy.linalg.norm(self.v)
        logger.info(f"self.x: {self.x*au2ev*1000} mev")
        logger.info(f"self.v**2: {self.v**2}")
        
        self.imps_sd = None
        if self.sd_algo in ["mps_sampling", "mps_sampling2"]:
            assert nsamples is not None
            self.nsamples = nsamples
            #prob = self.wfn**2
            #prob_sum = [0]
            #for i in range(len(self.wfn)):
            #    prob_sum.append(prob_sum[-1]+prob[i])
            #del prob_sum[0]
            #prob_sum = np.array(prob_sum)/prob_sum[-1]
            #logger.debug(f"prob:{prob},{np.sum(prob)}")
            #logger.debug(f"prob_sum: {prob_sum}")
            rng = np.random.default_rng()
            sample_index = rng.choice(len(self.x), size=(self.nsites_sd,
                nsamples), p=self.v**2)
            #sample = rng.uniform(size=(self.nsites_sd, nsamples))
            #sample_index = np.searchsorted(prob_sum, sample)
            self.unique_sample_index, self.counts = np.unique(sample_index,
                    axis=1, return_counts=True)
            logger.debug(f"unique_sample_index: {self.unique_sample_index}")
            logger.debug(f"sample counts: {self.counts}")
            #sigma = np.sqrt(1/2/self.w_sd)
            #de = rng.normal(0, sigma, (self.nsites_sd, nsamples))
            #diff = np.abs(de.reshape(1, self.nsites_sd, nsamples)-self.x.reshape(-1,1,1))
            #self.sample_index = np.argmin(diff, axis=0)
            self.imps_sample = None
        
        if self.sd_algo == "mps_sampling2" and imps is not None:
            logger.info(f"analyze mpdm")
            nmols = self.nmols
            identity = Mpo.identity(model)
            self.imps_sample = self._sample_mps_sd2(imps, identity, imps)
            e_rdm = np.zeros((nmols, nmols), dtype=np.complex128)
            for idx in range(nmols):
                for jdx in range(idx, nmols):
                    op = Op(r"a^\dagger a", [f"e_{idx}", f"e_{jdx}"])
                    mpo = Mpo(model, terms=op)
                    e_rdm_sample = self._sample_mps_sd2(imps, mpo, imps) 
                    e_rdm[idx, jdx] = np.sum(e_rdm_sample / self.imps_sample * self.counts) / self.nsamples 
                    e_rdm[jdx, idx] = np.conj(e_rdm[idx, jdx])
            
            # Jiang's JPCL paper
            L1 = np.sum(np.abs(e_rdm))**2 /  np.sum(np.abs(e_rdm)**2) / nmols
            L3 = np.trace(e_rdm @ e_rdm) * nmols
            np.save("e_rdm", e_rdm)
            logger.info(f"electronic coherence length: L1: {L1}, L3: {L3}")
            
            # calculate vibrational distortion field
            logger.info("Calculate vibrational distortion field")
            displacement = []
            nmodes =  len(self.w_intra)
            for imode in range(nmodes):
                logger.debug(f"mode index: {imode}")
                displacement.append([])
                for dis in range(-nmols+1,nmols):
                    ops = []
                    for imol in range(nmols):
                        if imol+dis >=0 and imol+dis<nmols:
                            op = Op("a^\dagger a x", [f"e_{imol}", f"e_{imol}",
                                f"v_{imol+dis}_{imode}"], factor=1, qn=[1,-1, 0])
                            ops.append(op)
                    mpo = Mpo(model, terms=ops)
                    res_sample = self._sample_mps_sd2(imps, mpo, imps) 
                    res = np.sum(res_sample / self.imps_sample * self.counts) / self.nsamples
                    displacement[imode].append(res)
            displacement = np.array(displacement).real
            np.save("intra_displacement", displacement)
            re_tot = 0.5*np.sum(self.w_intra**2*np.sum(displacement**2,axis=1))
            logger.info(f"total effective intramolecular reorganization energy: {re_tot}")
            
            # calculate expectation value of each mode
            logger.info("Calculate expectation value of each mode")
            
            q_mean = []
            for imol in range(nmols):
                for imode in range(nmodes):
                    logger.debug(f"mode index: {imol}, {imode}")
                    op = Op("x", [f"v_{imol}_{imode}"], factor=1, qn=[0])
                    mpo = Mpo(model, terms=[op])
                    res_sample = self._sample_mps_sd2(imps, mpo, imps) 
                    res = np.sum(res_sample / self.imps_sample * self.counts) / self.nsamples
                    q_mean.append(res)
            q_mean = np.array(q_mean).reshape(nmols,-1)
            np.save("intra_q_mean", q_mean)
            re_tot = 0.5*np.einsum("ab,b->", q_mean**2, self.w_intra**2)
            logger.info(f"total effective intramolecular reorganization energy by q_mean: {re_tot}")

            assert False

        super(FTCorrFuncHolsteinStaticDisorder, self).__init__(
                model,
                imps_qntot,
                temperature,
                insteps,
                imps = imps,
                ievolve_config=ievolve_config,
                icompress_config=icompress_config,
                compress_config=compress_config,
                evolve_config=evolve_config,
                optimize_config=optimize_config,
                dump_mps=dump_mps,
                dump_dir=dump_dir,
                job_name=job_name,
                )
        

    def wfn_sd(self, x):
        return (self.w_sd/np.pi)**0.25 * np.exp(-0.5*self.w_sd*x**2)
    
    def init_imps(self):
        if self.imps is not None:
            return self.imps

        mpdm = MpDm.max_entangled_mpdm(self.model, self.imps_qntot)
        
        # initialize the sd disorder site, it is not locally maximally-entangled
        for isite in range(self.nsites_sd):
            local_basis = self.model.basis[isite]
            assert isinstance(local_basis, ba.BasisSHODVR) or isinstance(local_basis, ba.BasisSineDVR)
            for i in range(local_basis.nbas):
                mpdm[isite][:,i,i,:] *= self.v[i]

        mpdm.compress_config = self.icompress_config

        tp = ThermalProp(
            mpdm, self.h_mpo, evolve_config=self.ievolve_config,
            dump_dir=self.dump_dir, job_name=self.job_name, include_ex=False
        )
        if tp._defined_output_path:
            try:
                logger.info(
                    f"load density matrix from {self._thermal_dump_path}"
                )
                mpdm = MpDm.load(self.model, self._thermal_dump_path)
                logger.info(f"density matrix loaded:{mpdm}")                
                return mpdm
            except FileNotFoundError:
                logger.debug(f"no file found in {self._thermal_dump_path}")
                logger.info(f"calculate mpdm from scratch.")
        
        tp.evolve(None, self.insteps, self.temperature.to_beta() / 2j)
        mpdm = tp.latest_mps
        if tp._defined_output_path:
            mpdm.dump(self._thermal_dump_path)
        return mpdm
    
    def _construct_mps_sd(self, mpdm):
        # constract site except the static disorder mode

        res = xp.eye(1)
        for ims in range(len(mpdm)-1, self.nsites_sd-1, -1):
            res = oe.contract("abbc,cd->ad", mpdm[ims], res, backend=OE_BACKEND)
        mpdm[self.nsites_sd-1] = oe.contract("abcd, de->abce",
                mpdm[self.nsites_sd-1], res)
        # extract the diagonal term
        mps_sd = []
        for ims in range(self.nsites_sd):
            shape = mpdm[ims].shape
            dtype = mpdm[ims].dtype
            ms = np.zeros((shape[0], shape[1], shape[3]), dtype=dtype)
            for i in range(shape[1]):
                ms[:,i,:] = mpdm[ims][:,i,i,:]
            mps_sd.append(ms)
        return mps_sd

    def _sample_mps_sd(self, mps_sd):
        res_sample = np.zeros(self.unique_sample_index.shape[1],dtype=np.complex128)
        for isample in range(self.unique_sample_index.shape[1]):
            res = xp.ones((1,1))
            for isite in range(self.nsites_sd):
                res = res @ asxp(mps_sd[isite][:,self.unique_sample_index[isite,isample],:])
            res_sample[isample] = res.item()
        return asnumpy(res_sample)
    
    def _sample_mps_sd2(self, bra, mpo, ket):
        res_sample = np.zeros(self.unique_sample_index.shape[1],dtype=np.complex128)
        
        environ = Environ(ket, mpo, "R", mps_conj=bra.conj())
        rtensor = environ.read("R", self.nsites_sd)
        for isample in range(self.unique_sample_index.shape[1]):
            bra_samp = xp.ones((1))
            ket_samp = xp.ones((1))
            mpo_samp = xp.ones((1))
            for ims in range(self.nsites_sd):
                idx = self.unique_sample_index[ims, isample]
                bra_samp = xp.tensordot(bra_samp, asxp(bra[ims][:,idx,idx,:]), axes=1)
                ket_samp = xp.tensordot(ket_samp, asxp(ket[ims][:,idx,idx,:]), axes=1)
                mpo_samp = xp.tensordot(mpo_samp, asxp(mpo[ims][:,idx,idx,:]), axes=1)  
            res_sample[isample] = oe.contract("a,c,e,ace->", bra_samp.conj(),
                    mpo_samp, ket_samp, rtensor, backend=OE_BACKEND).item()
        return asnumpy(res_sample)

    def process_mps(self, braket_pair):
        t = self.evolve_times[-1]
        self._autocorr_time.append(t)
        
        mpo, ket, bra = braket_pair.mpo, braket_pair.ket_mps, braket_pair.bra_mps
        if self.sd_algo == "mps_sampling2":
            if self.imps_sample is None:
                identity = Mpo.identity(self.model)
                self.imps_sample = self._sample_mps_sd2(self.imps, identity, self.imps)
            ft_sample = self._sample_mps_sd2(bra, mpo, ket) * np.exp(1.0j*t*(self.e_bra -\
                self.e_ket)) * np.conjugate(bra.coeff) * ket.coeff
            ft = np.sum(ft_sample / self.imps_sample * self.counts) / self.nsamples 
        else:
            mpdm = mpo @ ket @ bra.conj_trans()
            
            if self.imps_sd is None:
                local_basis = self.imps.model.basis[:self.nsites_sd]
                model = Model(local_basis, [])
                self.imps_sd = self._construct_mps_sd(self.imps @ self.imps.conj_trans())
                self.imps_sd = Mps.from_mp(model, self.imps_sd).canonicalise().canonicalise()
                if self.sd_algo == "mps_sampling":
                    # the sampling for the thermal eq state
                    self.imps_sample = self._sample_mps_sd(self.imps_sd)    
                elif self.sd_algo == "linear_equation":
                    self.imps_sd = Mpo.from_mps(self.imps_sd)
                else:
                    assert False

            mpdm_sd = self._construct_mps_sd(mpdm)
            mpdm_sd = Mps.from_mp(self.imps_sd.model, mpdm_sd).canonicalise().canonicalise()
            
            if self.sd_algo == "mps_sampling":
                ft_sample = self._sample_mps_sd(mpdm_sd) * np.exp(1.0j*t*(self.e_bra -\
                    self.e_ket)) * np.conjugate(bra.coeff) * ket.coeff
                ft = np.sum(ft_sample / self.imps_sample * self.counts) / self.nsamples 
            
            elif self.sd_algo == "linear_equation":
                ft_sd = Mps.random(self.imps_sd.model, 0, 10).to_complex()
                ft_sd.optimize_config = self.optimize_config
                _, ft_sd = solve_mps(ft_sd, mpdm_sd, self.imps_sd, mpo_kind="her")
                for ms in ft_sd:
                    for ibas in range(ms.shape[1]):
                        ms[:,ibas,:] *= self.v[ibas]**2
                mode = ["all"] * self.nsites_sd
                ft = ft_sd.contract_self(mode).item()
                ft *= np.exp(1.0j*t*(self.e_bra -\
                    self.e_ket)) * np.conjugate(bra.coeff) * ket.coeff * ft_sd.coeff
            else:
                assert False

        logger.info(f"ft with static disorder: {ft}")
        self._autocorr.append(ft)

class ZTCorrFuncBase(CorrFuncBase):
    r'''Abstract class 
    '''
    
    def init_imps(self):
        if self.imps is not None:
            return self.imps
        m_init = 20
        imps = Mps.random(self.model, self.imps_qntot, m_init, percent=1)
        imps.optimize_config = self.optimize_config
        energies, imps = gs.optimize_mps(imps, self.h_mpo)
        #e_rdm = imps.calc_reduced_density_matrix()
        #logger.info(f"e_rdm: {e_rdm}")
        return imps

    def process_mps(self, braket_pair):
        if not self.evolve_times[-1] == 0:
            last_bra_mps, last_ket_mps = self.latest_mps
            bra_mps, ket_mps = braket_pair
            t_ket = self.evolve_times[-1] 
            t_bra = self.evolve_times[-2]
            self._autocorr.append(BraKetPair(last_bra_mps, ket_mps).ft *
                    np.exp(-1.0j*t_ket*self.e_ket) *
                    np.exp(-1.0j*t_bra*self.e_bra) *
                    np.exp(1.0j*(t_bra+t_ket)*self.e_imps))
            self._autocorr_time.append(t_bra + t_ket)
            
        t = self.evolve_times[-1]
        self._autocorr.append(braket_pair.ft *
                np.exp(-1.0j*t*(self.e_ket+self.e_bra)) *
                np.exp(1.0j*2*t*self.e_imps))
        self._autocorr_time.append(2*t)
        
        bra_mps, ket_mps = braket_pair
        #bra_rdm = bra_mps.calc_reduced_density_matrix()
        #ket_rdm = ket_mps.calc_reduced_density_matrix()
        #self.e_rdm.append([bra_rdm, ket_rdm])

class ZTCorrFuncBase_state_to_state(ZTCorrFuncBase):
    def __init__(self, broad_func, acf_stos_compress_config, *args, **kw):
        # broadening function
        self.broad_func = broad_func
        assert acf_stos_compress_config is not None
        self.acf_stos_compress_config = acf_stos_compress_config
        self.acf_stos = None
        self.rate = [0,]
        super().__init__(*args, **kw)

    def process_mps(self, braket_pair):
        bra_mps, ket_mps = braket_pair
        
        mpdm_compress_config = self.acf_stos_compress_config

        if not self.evolve_times[-1] == 0:
            last_bra_mps, last_ket_mps = self.latest_mps
            t_ket = self.evolve_times[-1] 
            t_bra = self.evolve_times[-2]

            mpdm_odd = MpDm.from_bra_ket(ket_mps, last_bra_mps.conj())
            mpdm_odd = mpdm_odd.scale(
                    np.exp(-1.0j*t_ket*self.e_ket) *
                    np.exp(-1.0j*t_bra*self.e_bra) *
                    np.exp(1.0j*(t_bra+t_ket)*self.e_imps)*self.broad_func(t_bra+t_ket))
            self._autocorr_time.append(t_bra + t_ket)
            mpdm_odd = mpdm_odd.diagonal()
            self._autocorr.append(mpdm_odd.sum())
            mpdm_odd.compress_config = mpdm_compress_config 
            mpdm_odd.canonicalise().compress()
         
        t = self.evolve_times[-1]
        mpdm_even = MpDm.from_bra_ket(ket_mps, bra_mps.conj())
        mpdm_even = mpdm_even.scale(
                np.exp(-1.0j*t*self.e_ket) *
                np.exp(-1.0j*t*self.e_bra) *
                np.exp(1.0j*2*t*self.e_imps)*self.broad_func(2*t))
        self._autocorr_time.append(2*t)
        mpdm_even = mpdm_even.diagonal()
        self._autocorr.append(mpdm_even.sum())
        mpdm_even.compress_config = mpdm_compress_config 
        mpdm_even.canonicalise().compress()
        
        if self.autocorr_time[-1] == 0:
            # the 0 evolve step
            self.acf_stos = mpdm_even
        else:
            dt = self.autocorr_time[-1] - self.autocorr_time[-2]
            if self.autocorr_time[-3] == 0:
                # the first evolve step
                new_acf_stos = self.acf_stos.scale(dt/2) + mpdm_odd.scale(dt) + mpdm_even.scale(dt)
            else:
                new_acf_stos = self.acf_stos + mpdm_odd.scale(dt) + mpdm_even.scale(dt)
            new_acf_stos.compress_config = mpdm_compress_config 
            new_acf_stos.canonicalise().compress()
            self.acf_stos = new_acf_stos
            current_rate = self.acf_stos.sum()
            self.rate.extend([current_rate,]*2)
        logger.debug(f"acf_stos:{self.acf_stos}")
        #bra_rdm = bra_mps.calc_reduced_density_matrix()
        #ket_rdm = ket_mps.calc_reduced_density_matrix()
        #self.e_rdm.append([bra_rdm, ket_rdm])

        
    def dump_dict(self):
        super().dump_dict()
        if self.acf_stos is not None:
            mps_path = os.path.join(self.dump_dir,
                    self.job_name+"_acf_stos" + ".npz")
            self.acf_stos.dump(mps_path)
    
    def get_dump_dict(self):
        super_dump_dict = super().get_dump_dict()
        super_dump_dict["rate"] = self.rate
        
        return super_dump_dict

class FTCorrFuncBase_state_to_state(FTCorrFuncBase):
    def __init__(self, broad_func, acf_stos_compress_config, *args, **kw):
        self.broad_func = broad_func
        assert acf_stos_compress_config is not None
        self.acf_stos_compress_config = acf_stos_compress_config
        self.acf_stos = None
        self.rate = [0,]
        super().__init__(*args, **kw)
    

    def process_mps(self, braket_pair):
        t = self.evolve_times[-1]
        bra_mps, ket_mps = braket_pair
        mpdm_compress_config = self.acf_stos_compress_config
        
        bra_mps = self.op_a.apply(bra_mps)
        bra_mps.canonicalise()
        bra_mps.compress()

        mpdm = ket_mps.apply(bra_mps.conj_trans()).scale(
            np.exp(1.0j*t*(self.e_bra - self.e_ket)) * self.broad_func(t))
        mpdm = mpdm.diagonal()      
        self._autocorr_time.append(t)
        self._autocorr.append(mpdm.sum())
        mpdm.compress_config = mpdm_compress_config
        logger.debug(f"mpdm:{mpdm}")
        mpdm.canonicalise()
        mpdm.compress()
        
        if self.autocorr_time[-1] == 0:
            self.acf_stos = mpdm
            # the 0 evolve step
        else:
            dt = self.autocorr_time[-1] - self.autocorr_time[-2]
            if self.autocorr_time[-2] == 0:
                # the first evolve step
                new_acf_stos = self.acf_stos.scale(dt/2) + mpdm.scale(dt) 
            else:
                new_acf_stos = self.acf_stos + mpdm.scale(dt)
            new_acf_stos.compress_config = mpdm_compress_config 
            new_acf_stos.canonicalise().compress()
            self.acf_stos = new_acf_stos
            self.rate.append(self.acf_stos.sum())
        logger.debug(f"acf_stos:{self.acf_stos}")

        #bra_rdm = bra_mps.calc_reduced_density_matrix()
        #ket_rdm = ket_mps.calc_reduced_density_matrix()
        #self.e_rdm.append([bra_rdm, ket_rdm])
    
    def dump_dict(self):
        super().dump_dict()
        if self.acf_stos is not None:
            mps_path = os.path.join(self.dump_dir,
                    self.job_name+"_acf_stos" + ".npz")
            self.acf_stos.dump(mps_path)
    
    def get_dump_dict(self):
        super_dump_dict = super().get_dump_dict()
        super_dump_dict["rate"] = self.rate
        
        return super_dump_dict

def analysis_dominant_config(mps, nconfigs=1):
    # analysis the dominant configuration an mps
    # mps should be real

    config_visited = []
    ci_coeff_list = []
    while len(config_visited) < nconfigs:
        mps_rank1 = mps.canonicalise().compress(temp_m_trunc=1)
        # get config with the largest coeff
        config = []
        for ims, ms in enumerate(mps_rank1):
            ms = ms.array.flatten()**2
            quanta = int(np.argmax(ms))
            config.append(quanta)
        
        while config in config_visited:
            # random mutate
            idx = np.random.randint(mps.site_num)
            quant = np.random.randint(mps.model.pbond_list[idx])
            config[idx] = quant

        config_visited.append(config)
        
        sentinel = xp.ones(1)
        for ims, ms in enumerate(mps):
            sentinel = xp.tensordot(sentinel, asxp(ms[:,config[ims],:]),
                    axes=(0,0))

        ci_coeff_list.append(sentinel.item()*mps.coeff)
        condition = {}
        for idx in range(len(config)):
            dofname = mps.model.basis[idx].dofs
            condition[dofname[0]] = config[idx]
        mps = mps - Mps.hartree_product_state(mps.model, condition).scale(ci_coeff_list[-1])

    return config_visited, ci_coeff_list   

def monte_carlo_sampling(mps, vec, prob=None, nconfigs=1, converge=None, bag=None):
    
    if converge is None:
        converge = (mps.sum()/mps.coeff).real

    np.random.seed()
    if bag is None:
        bag = dict()
        tot = 0
    else:
        assert isinstance(bag, dict)
        tot = np.sum(list(bag.values())).real
        logger.info(f"current tot: {tot}")

    coeff = ci_coeff(mps, vec)
    if vec not in bag:
        bag[vec] = coeff
        tot += coeff.real

    while len(bag) < nconfigs and tot < converge:
        isite = np.random.randint(mps.site_num)
        if prob is None:
            quant = np.random.randint(mps.model.pbond_list[isite])
        else:
            p = np.random.random()
            for ival, val in enumerate(prob[isite]):
                if p < val:
                    break
            quant = ival
        # if not mutated, continue
        if quant == vec[isite]:
            continue
        vec_new = list(vec)
        vec_new[isite] = quant
        vec_new = tuple(vec_new)
        if vec_new in bag:
            coeff_new = bag[vec_new]
        else:
            coeff_new = ci_coeff(mps, vec_new)
            bag[vec_new] = coeff_new
            tot += coeff_new.real
        
        # if accept the mutation 
        p = min(1, coeff_new.real/coeff.real)
        if np.random.random() < p:
            vec = vec_new
            coeff = coeff_new
    return bag

def independent_sample(mps, prob, nconfigs=1, thresh=0):
    np.random.seed()
    bag = dict()
    while len(bag) < nconfigs:
        p = np.random.random(size=mps.site_num)
        config = []
        for isite in range(mps.site_num):
            for ival, val in enumerate(prob[isite]):
                if p[isite] < val:
                    break
            config.append(ival)
        config = tuple(config)
        if config in bag:
            continue
        coeff = ci_coeff(mps, config)
        if coeff.real > thresh:
            bag[config] = coeff
    return bag

def ci_coeff(mps, config):
    sentinel = xp.ones(1)
    for ims, ms in enumerate(mps):
        sentinel = xp.tensordot(sentinel, asxp(ms[:,config[ims],:]),
                axes=(0,0))
    return sentinel.item()


class ZTAACorrFuncBase(ZTCorrFuncBase):
    r''' Abstract class
    Note: please make sure the operator A is real
    '''
    def init_mps(self):
        ket_mps = self.op_b.apply(self.imps)
        ket_mps.normalize("mps_norm_to_coeff")
        
        bra_mps = ket_mps.copy()

        return self.pruner(BraKetPair(bra_mps, ket_mps))

    def init_op_a(self):
        return self.init_op_b()
    
    def evolve_single_step(self, evolve_dt):
        _, ket_mps = self.latest_mps
        ket_mps = ket_mps.evolve(self.ket_mpo, evolve_dt)
        bra_mps = ket_mps.conj()
        return BraKetPair(bra_mps, ket_mps)
        
class ZTAACorrFuncBase_state_to_state(ZTAACorrFuncBase,
        ZTCorrFuncBase_state_to_state):
    pass

class ZTABCorrFuncBase(ZTCorrFuncBase):
    r'''Abstract class 
    '''
    
    def init_mps(self):
        ket_mps = self.op_b.apply(self.imps)
        ket_mps.normalize("mps_norm_to_coeff")
        bra_mps = self.op_a.apply(self.imps)
        bra_mps.normalize("mps_norm_to_coeff")
        return self.pruner(BraKetPair(bra_mps, ket_mps))
    
    def evolve_single_step(self, evolve_dt):
        bra_mps, ket_mps = self.latest_mps
        ket_mps = ket_mps.evolve(self.ket_mpo, evolve_dt)
        bra_mps = bra_mps.evolve(self.bra_mpo, -evolve_dt)
        return BraKetPair(bra_mps, ket_mps)
    

class ZTabs(ZTAACorrFuncBase):
    
    def init_op_b(self):
        return abs_dipole_op(self.model)
    
    def init_imps(self):
        assert self.imps_qntot == 0
        return super().init_imps()

class ZTabs_TFD(ZTAACorrFuncBase):
    
    def init_op_b(self):
        return abs_dipole_op(self.model)

    def init_imps(self):
        assert self.imps_qntot == 0
        e_dofs = self.model.e_dofs
        for dof in e_dofs:
            siteidx = self.model.dof_to_siteidx[dof]
            local_basis = self.model.basis[siteidx]
            idx = local_basis.dof_name_map[dof]
            if local_basis.sigmaqn[idx] == 0:
                init_condition = {dof:idx}
                break

        imps = Mps.hartree_product_state(self.model, condition=init_condition)
        return imps


class ZTemi(ZTAACorrFuncBase):

    def init_op_b(self):
        return emi_dipole_op(self.model)

    def init_imps(self):
        assert self.imps_qntot == 1
        return super().init_imps()


class FTabs(FTCorrFuncBase):
    
    def init_op_b(self):
        return abs_dipole_op(self.model)
    
    def init_op_a(self):
        return self.init_op_b()
    
    def init_imps(self):
        assert self.imps_qntot == 0
        return super().init_imps()

class FTabs_sd(FTCorrFuncHolsteinStaticDisorder, FTabs):
    pass

class FTemi(FTCorrFuncBase):

    def init_op_b(self):
        return emi_dipole_op(self.model)
    
    def init_op_a(self):
        return self.init_op_b()

    def init_imps(self):
        assert self.imps_qntot == 1
        return super().init_imps()

class FTemi_sd(FTCorrFuncHolsteinStaticDisorder, FTemi):
    pass

class ZTnr(ZTAACorrFuncBase):

    def init_op_b(self):
        return nac_op(self.model)

    def init_imps(self):
        assert self.imps_qntot == 1
        return super().init_imps()

class ZTnr_state_to_state(ZTAACorrFuncBase_state_to_state):
    
    def init_op_b(self):
        return nac_op(self.model)

    def init_imps(self):
        assert self.imps_qntot == 1
        return super().init_imps()

class FTnr(FTCorrFuncBase):

    def init_op_b(self):
        return nac_op(self.model)

    def init_op_a(self):
        return self.init_op_b()

    def init_imps(self):
        assert self.imps_qntot == 1
        return super().init_imps()

class FTnr_sd(FTCorrFuncHolsteinStaticDisorder, FTnr):
    pass
    
class FTnr_state_to_state(FTCorrFuncBase_state_to_state):
    
    def init_op_b(self):
        return nac_op(self.model)

    def init_op_a(self):
        return self.init_op_b()

    def init_imps(self):
        assert self.imps_qntot == 1
        return super().init_imps()

###################################
# ic + ct coupling Xiankai Chen
class ZTic_ct(ZTAACorrFuncBase):
    def init_op_b(self):
        return ic_ct_op(self.model)

    def init_imps(self):
        assert self.imps_qntot == 1
        return super().init_imps()

class FTic_ct(FTCorrFuncBase):

    def init_op_b(self):
        return ic_ct_op(self.model)

    def init_op_a(self):
        return self.init_op_b()

    def init_imps(self):
        assert self.imps_qntot == 1
        return super().init_imps()

def ic_ct_op(model):
    # the momentum operator coupling + ct coupling
    # the quantum number is gs: 0 ex:1
    if "ic_ct_mpo" in model.mpos.keys():
        logger.info("load ic_ct_mpo form model.mpos")
        return model.mpos["ic_ct_mpo"]
    else:
        h_prime_terms = []
        for key, value in model.para["nac"].items():
            siteidx = model.dof_to_siteidx[key[0]]
            local_basis = model.basis[siteidx]
            assert isinstance(local_basis,ba.BasisMultiElectron)
            idx0 = local_basis.dof.index(key[0])
            idx1 = local_basis.dof.index(key[1])
            assert local_basis.sigmaqn[idx0] == 0 and local_basis.sigmaqn[idx1] == 1
            h_prime_terms.append(Op("a^\dagger a partialx", list(key),
                factor=-value, qn=[0,-1,0]))    
        for key, value in model.para["t_ct_gs"].items():
            siteidx = model.dof_to_siteidx[key[0]]
            local_basis = model.basis[siteidx]
            assert isinstance(local_basis,ba.BasisMultiElectron)
            idx0 = local_basis.dof.index(key[0])
            idx1 = local_basis.dof.index(key[1])
            assert local_basis.sigmaqn[idx0] == 0 and local_basis.sigmaqn[idx1] == 1
            h_prime_terms.append(Op("a^\dagger a", list(key),
                factor=value, qn=[0,-1]))    

        return Mpo(model, terms=h_prime_terms)



class FTcurrent_current_Holstein(FTCorrFuncBase):
    def init_op_b(self):
        j_op, _ = current_op(self.model, None)
        return j_op

    def init_op_a(self):
        return self.init_op_b()

    def init_imps(self):
        assert self.imps_qntot == 1
        return super().init_imps()

class FTnr(FTCorrFuncBase):

    def init_op_b(self):
        return nac_op(self.model)

    def init_op_a(self):
        return self.init_op_b()

    def init_imps(self):
        assert self.imps_qntot == 1
        return super().init_imps()

# isc rate
#################################################################3
def isc_op(model):
    isc_terms = []
    for key, value in model.para["isc"].items():
        # the singlet state has qn=1, triplet state has qn=0
        op = Op("a^\dagger a", list(key), factor=value, qn=[0,-1])
        isc_terms.append(op)
    return Mpo(model, terms=isc_terms)

class ZTisc(ZTAACorrFuncBase):

    def init_op_b(self):
        return isc_op(self.model)

    def init_imps(self):
        assert self.imps_qntot == 1
        return super().init_imps()

class FTisc(FTCorrFuncBase):

    def init_op_b(self):
        return isc_op(self.model)

    def init_op_a(self):
        return self.init_op_b()

    def init_imps(self):
        assert self.imps_qntot == 1
        return super().init_imps()
################################################################

def risc_op(model):
    raise NotImplementedError

class ZT_sbm_spin_spin(ZTAACorrFuncBase):

    def init_op_b(self):
        return spin_op(self.model)

    def init_imps(self):
        assert self.imps_qntot == 0
        return super().init_imps()

class FT_sbm_spin_spin(FTCorrFuncBase):

    def init_op_b(self):
        return spin_op(self.model)
    
    def init_op_a(self):
        return self.init_op_b()

    def init_imps(self):
        assert self.imps_qntot == 0
        return super().init_imps()

class sbm_spin(ZTABCorrFuncBase):

    def init_op_a(self):
        return spin_op(self.model)
    
    def init_op_b(self):
        return Mpo.identity(self.model)

    def init_imps(self):
        assert self.imps_qntot == 0
        return super().init_imps()

################################################################
class ZTIRAA(ZTAACorrFuncBase):
    
    def init_op_b(self):
        return IR_op(self.model,"B")
    
    def init_imps(self):
        assert self.imps_qntot == 0
        return super().init_imps()

class FTIRAA(FTCorrFuncBase):
    
    def init_op_b(self):
        return IR_op(self.model,"B")
    
    def init_op_a(self):
        return self.init_op_b()
    
    def init_imps(self):
        assert self.imps_qntot == 0
        return super().init_imps()

class ZTIRAB(ZTABCorrFuncBase):
    
    def init_op_b(self):
        return IR_op(self.model, "B")
    
    def init_op_a(self):
        return IR_op(self.model, "A")
    
    def init_imps(self):
        assert self.imps_qntot == 0
        return super().init_imps()

def IR_op(model, op_symbol):
    return model.mpos["IR"+op_symbol]

#class PhotoPhysics(CorFuncTdMpsJobBase):
#    r""" Photophysics properties of molecular aggregates with or without DRE
#    """
#    def __init__(
#            self,
#            model,
#            job_type,
#            temperature,
#            optimize_config=None,
#            evolve_config=None,
#            dump_dir=None,
#            job_name=None,
#            ):
#        
#        if optimize_config is None:
