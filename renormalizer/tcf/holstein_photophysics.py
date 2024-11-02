from renormalizer.tcf import base
from renormalizer.mps import MpDm, Mpo
from renormalizer.model import Model, Op
from renormalizer.utils.constant import *
from renormalizer.model import basis as ba
from renormalizer.utils import EvolveConfig, OptimizeConfig, CompressConfig, CompressCriteria, EvolveMethod
from renormalizer.utils import log, Quantity
from renormalizer.utils.filter import modes_filter
from renormalizer.tcf.fft import *

import logging
import itertools 
import numpy as np

logger = logging.getLogger("renormalizer")

def holstein_photophysics(
                    hsys, 
                    photophysics_type,
                    dipole,
                    temperature,
                    nbas, 
                    bonddim,
                    nsteps, 
                    dt, 
                    from_momap=None, # [fdusin, fnac, lamb_thresh, nac_thresh, pes_reference]
                    from_user_define=None, # [w, lamb, nac]
                    inter_vib=None,
                    thermal_impo=None,
                    sd=None,  # Normal distribution of static disorder
                    sd_algo="mps_sampling2", # "direct_sampling", "mps_sampling" or "linear_equation"
                    nsamples=500,
                    nbas_sd=None,
                    optimize_config=None,
                    dvr="SHODVR",
                    seed=None,
                    ):
    
    if optimize_config is None:
        vconfig = CompressConfig(CompressCriteria.threshold,
                threshold=1e-6)
        procedure = [[vconfig,0.5], [vconfig,0.3], [vconfig, 0.1]] + [[vconfig, 0]]*20
        optimize_config = OptimizeConfig(procedure)

    dump_dir = "./"
    job_name = "result" 
    log.register_file_output(dump_dir+job_name+".log", mode="w")
    
    if from_momap is not None:
        fdusin, fnac, lamb_thresh, nac_thresh, pes_reference = from_momap

        # get the parameters
        w0, w1, d0, d1, nac, s021, s120 = base.single_mol_model(fdusin, fnac, projector=6)
        if pes_reference == 0:
            w, d = w0, d0
        elif pes_reference == 1:
            w, d = w1, d1
        else:
            assert False

        logger.info("Diplaced harmonic oscillator model is used!")
        logger.info(f"w: {w*au2cm}")
        logger.info(f"original nmodes: {len(w)}")
        lamb = 1/2*w**2*d**2
        logger.info(f"lambda: {lamb*au2cm}")
        reorg_e = np.sum(lamb)
        logger.info(f"original reorg_e: {reorg_e}")
        
        if photophysics_type == "ic":
            assert fnac is not None
            assert pes_reference == 1
            w, lamb, nac = modes_filter(w, lamb, lamb_thresh, nac, nac_thresh)
        else:
            w, lamb, nac = modes_filter(w, lamb, lamb_thresh)
        
        index = np.argsort(lamb)[::-1]
        w, lamb = w[index], lamb[index]
        if photophysics_type == "ic":
            nac = nac[index]

        logger.info(f"nmodes after filter: {len(w)}")
        logger.info(f"w after filter: {w*au2cm}")
        logger.info(f"lambda after filter: {lamb*au2cm}")
        assert np.allclose(np.sum(lamb), reorg_e)
    else:
        assert from_user_define is not None
        w, lamb, nac = from_user_define
        reorg_e = np.sum(lamb)

    nmodes = len(w)
    nmols = hsys.shape[0]
    
    # construct the model
    ham_terms = []
    for imol in range(nmols):
        for imode in range(nmodes):
            ham_terms.append(Op("p^2", f"v_{imol}_{imode}", factor=1/2, qn=0))
            ham_terms.append(Op("x^2", f"v_{imol}_{imode}", factor=1/2*w[imode]**2, qn=0))
    
    for imol in range(nmols):
        for jmol in range(nmols):
            factor = hsys[imol,jmol] 
            if imol == jmol:
                factor += reorg_e

            ham_terms.append(Op(r"a^\dagger a", [f"e_{imol}", f"e_{jmol}"],
                factor=factor, qn=[1,-1]))
    
    for imol in range(nmols):
        for imode in range(nmodes):
            ham_terms.append(Op(r"a^\dagger a x", [f"e_{imol}", f"e_{imol}",
                f"v_{imol}_{imode}"], 
                factor = np.sqrt(2*lamb[imode]*w[imode]**2), qn=[1,-1,0]))
    
    if inter_vib is not None: 
        inter_w, inter_g, inter_nbas, link = inter_vib
        for imol, jmol in link:
            ham_terms.append(Op("p^2", f"inter_v_{imol}_{jmol}", factor=1/2, qn=0))
            ham_terms.append(Op("x^2", f"inter_v_{imol}_{jmol}", factor=1/2*inter_w**2, qn=0))
            ham_terms.append(Op(r"a^\dagger a x", [f"e_{imol}", f"e_{jmol}",
                f"inter_v_{imol}_{jmol}"], factor = inter_g * inter_w * np.sqrt(2*inter_w), qn=[1,-1,0]))
            ham_terms.append(Op(r"a^\dagger a x", [f"e_{jmol}", f"e_{imol}",
                f"inter_v_{imol}_{jmol}"], factor = inter_g * inter_w * np.sqrt(2*inter_w), qn=[1,-1,0]))
    
    if sd is not None:
        mu, sigma = 0, sd

        if sd_algo == "direct_sampling":
            # direct sampling the site energy
            rng = np.random.default_rng(seed=seed)
            de = rng.normal(mu, sigma, nmols)
            logger.info(f"static disorder de:{de}")
            for imol in range(nmols):
                ham_terms.append(Op(r"a^\dagger a", [f"e_{imol}", f"e_{imol}"],
                    factor=de[imol], qn=[1,-1]))
        else:
            assert temperature != 0
            w_sd = 1/2/sigma**2
            for imol in range(nmols):
                ham_terms.append(Op(r"a^\dagger a x", [f"e_{imol}", f"e_{imol}",
                    f"v_{imol}_sd"], factor=1, qn=[1,-1, 0]))
        
    basis = []
    if sd is not None and sd_algo != "direct_sampling":
        if nbas_sd is None:
            nbas_sd = nbas
        for imol in range(nmols):
            if dvr == "SHODVR":
                basis.append(ba.BasisSHODVR(f"v_{imol}_sd", w_sd,
                    nbas=nbas_sd))
            elif dvr == "SineDVR":
                basis.append(ba.BasisSineDVR(f"v_{imol}_sd",  nbas_sd, -5*sigma, 5*sigma))
            else:
                assert False

    basis.append(ba.BasisMultiElectron(["gs"]+[f"e_{imol}" for imol in range(nmols)], [0]+[1,]*nmols))
    
    if inter_vib is not None:
        for imol, jmol in link:
            basis.append(ba.BasisSHO(f"inter_v_{imol}_{jmol}", inter_w, inter_nbas))

    for imode in range(nmodes):
        for imol in range(nmols):
            basis.append(ba.BasisSHO(f"v_{imol}_{imode}", w[imode], nbas))
    
    
    # dipole moment and nac operator
    if photophysics_type != "ic":
        para = {"dipole":{}}   
        for imol in range(nmols):                              
            para["dipole"][(f"e_{imol}","gs")] = dipole
    else:
        para = {"nac":{}}
        for imol in range(nmols):
            for imode in range(nmodes):
                para["nac"][("gs", f"e_{imol}", f"v_{imol}_{imode}")] = nac[imode]


    model = Model(basis, ham_terms, para=para)
    
    imps = None
    if thermal_impo is not None:
        logger.info(
            f"load density matrix result_impo.npz"
        )
        mpdm = MpDm.load(model, thermal_impo)
        imps = mpdm
        logger.info(f"density matrix loaded:{mpdm}")
        if sd is None:
            e_rdm = mpdm.calc_edof_rdm()
            e_rdm = e_rdm[1:,1:]
            # Jiang's JPCL paper
            L1 = np.sum(np.abs(e_rdm))**2 /  np.sum(np.abs(e_rdm)**2) / nmols
            L3 = np.trace(e_rdm @ e_rdm) * nmols
            np.save("e_rdm", e_rdm)
            logger.info(f"electronic coherence length: L1: {L1}, L3: {L3}")

            # calculate vibrational distortion field
            logger.info("Calculate vibrational distortion field")
            displacement = []
            for imode in range(nmodes):
                displacement.append([])
                for dis in range(-nmols+1,nmols):
                    ops = []
                    for imol in range(nmols):
                        if imol+dis >=0 and imol+dis<nmols:
                            op = Op(r"a^\dagger a x", [f"e_{imol}", f"e_{imol}",
                                f"v_{imol+dis}_{imode}"], factor=1, qn=[1,-1, 0])
                            ops.append(op)
                    mpo = Mpo(model, terms=ops)
                    res = mpdm.expectation(mpo)
                    displacement[imode].append(res)
            displacement = np.array(displacement).real
            np.save("intra_displacement", displacement)
            re_tot = 0.5*np.sum(w**2*np.sum(displacement**2,axis=1))
            logger.info(f"total effective intramolecular reorganization energy by VDF: {re_tot}")
            
            if inter_vib is not None:
                for imol, jmol in link:
                    assert jmol == imol + 1
                
                displacement = []
                for dis in range(-nmols+1,nmols-1):
                    ops = []
                    for imol in range(nmols):
                        if imol+dis >=0 and imol+dis < nmols-1:
                            op = Op(r"a^\dagger a x", [f"e_{imol}", f"e_{imol}",
                                f"inter_v_{imol+dis}_{imol+dis+1}"], factor=1, qn=[1,-1, 0])
                            ops.append(op)
                    mpo = Mpo(model, terms=ops)
                    res = mpdm.expectation(mpo)
                    displacement.append(res)
                displacement = np.array(displacement).real
                np.save("inter_displacement", displacement)
                re_tot = 0.5*inter_w**2*np.sum(displacement**2)
                logger.info(f"total effective intermolecular reorganization energy by VDF: {re_tot}")

            
            # calculate expectation value of each mode
            logger.info("Calculate expectation value of each mode")
            mpos = []
            for imol in range(nmols):
                for imode in range(nmodes):
                    op = Op("x", [f"v_{imol}_{imode}"], factor=1, qn=[0])
                    mpo = Mpo(model, terms=[op])
                    mpos.append(mpo)
            q_mean = mpdm.expectations(mpos).reshape(nmols,-1)
            np.save("intra_q_mean", q_mean)
            re_tot = 0.5*np.einsum("ab,b->", q_mean**2, w**2)
            logger.info(f"total effective intramolecular reorganization energy by q_mean: {re_tot}")
            
            if inter_vib is not None:
                mpos = []
                for imol, jmol in link:
                    op = Op("x", [f"inter_v_{imol}_{jmol}"], factor=1, qn=[0])
                    mpo = Mpo(model, terms=[op])
                    mpos.append(mpo)
                res = mpdm.expectations(mpos)
                np.save("inter_q_mean", res)
                re_tot = 0.5*inter_w**2*np.sum(res**2)
                logger.info(f"total effective intermolecular reorganization energy by q_mean: {re_tot}")

            return None

    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps,
            adaptive=True,
            guess_dt=1e-2*fs2au,
            adaptive_rtol=1e-4,
            )
    compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=bonddim)
    
    if temperature == 0:
        if photophysics_type == "emi":
            kernel = base.ZTemi
            imps_qntot = 1
        elif photophysics_type == "abs":
            kernel = base.ZTabs
            imps_qntot = 0
        elif photophysics_type == "ic":
            kernel = base.ZTnr
            imps_qntot = 1
        
        job = kernel(
                     model,
                     imps_qntot,
                     Quantity(temperature,"K"),
                     optimize_config=optimize_config,
                     compress_config=compress_config,
                     evolve_config=evolve_config,
                     dump_mps=None,
                     dump_dir=dump_dir,
                     job_name=job_name)

    else:
        if photophysics_type == "emi":
            if sd is not None and sd_algo != "direct_sampling":
                kernel = base.FTemi_sd
            else:
                kernel = base.FTemi
            imps_qntot = 1

        elif photophysics_type == "abs":
            if sd is not None and sd_algo != "direct_sampling":
                kernel = base.FTabs_sd
            else:
                kernel = base.FTabs
            imps_qntot = 0
        elif photophysics_type == "ic":
            if sd is not None and sd_algo != "direct_sampling":
                kernel = base.FTnr_sd
            else:
                kernel = base.FTnr
            imps_qntot = 1
        beta = 1 / (temperature * K2au)
        ievolve_config = EvolveConfig(EvolveMethod.tdvp_ps,
                adaptive=True,
                guess_dt=beta/1j/2/1000,
                adaptive_rtol=1e-4,
                normalize="mps_and_coeff",
                )
        insteps = 20

        if sd is not None and sd_algo != "direct_sampling":
            job = kernel(
                         model,
                         imps_qntot,
                         Quantity(temperature, "K"),
                         insteps,
                         nmols,
                         w,
                         w_sd,
                         sd_algo,
                         imps=imps,
                         nsamples=nsamples,
                         optimize_config=optimize_config,
                         compress_config=compress_config,
                         evolve_config=evolve_config,
                         icompress_config=compress_config,
                         ievolve_config=ievolve_config,
                         dump_mps=None,
                         dump_dir=dump_dir,
                         job_name=job_name)
        else:
            job = kernel(
                         model,
                         imps_qntot,
                         Quantity(temperature, "K"),
                         insteps,
                         compress_config=compress_config,
                         evolve_config=evolve_config,
                         icompress_config=compress_config,
                         ievolve_config=ievolve_config,
                         dump_mps=None,
                         dump_dir=dump_dir,
                         job_name=job_name)


    
    job.evolve(dt, nsteps)
    ct = np.array(job._autocorr)
    t = np.array(job._autocorr_time)
    
    tau = t[-1] / 2
    logger.info(f"broaden constant in time/fs: {tau*au2fs}")
    broaden_constant = 1/tau
    logger.info(f"broaden constant in energy/ev: {broaden_constant*au2ev}")

    if photophysics_type == "abs":
        ct0, t0, cw0, w0 = ct2cw((ct, t), broaden_constant=None, fft_type="backward")
        ct, t, cw, w = ct2cw((ct, t), broaden_constant=broaden_constant, fft_type="backward")
        spectrum0 = np.stack((w0,cw0.real*w0*coeff_abs), axis=1)
        spectrum = np.stack((w,cw.real*w*coeff_abs), axis=1)
    elif photophysics_type == "emi":
        logger.info("without broadening:")
        ct0, t0, cw0, w0 = ct2cw((ct, t), broaden_constant=None, fft_type="forward")
        cumu_rate0, simpson_rate0 = emi_rate(cw0, w0)
        logger.info("with broadening:")
        ct, t, cw, w = ct2cw((ct, t), broaden_constant=broaden_constant, fft_type="forward")
        cumu_rate, simpson_rate = emi_rate(cw, w)
        np.save("cumu_rate", np.stack((t,cumu_rate),axis=1))
        spectrum0 = np.stack((w0,cw0.real*w0**3*coeff_emi), axis=1)
        spectrum = np.stack((w,cw.real*w**3*coeff_emi), axis=1)
    elif photophysics_type == "ic":
        logger.info("without broadening:")
        ct0, t0, cw0, w0, cumu_rate, simpson_rate = ct2cw((ct, t),
                broaden_constant=None, fft_type="backward", ic_rate=True)
        spectrum0 = np.stack((w0,cw0.real), axis=1)
        logger.info("with broadening:")
        ct, t, cw, w, cumu_rate, simpson_rate = ct2cw((ct, t), broaden_constant=broaden_constant,
                fft_type="backward", ic_rate=True)
        np.save("cumu_rate", np.stack((w,cumu_rate),axis=1))
        spectrum = np.stack((w,cw.real), axis=1)
    
    np.save("spectrum0", spectrum0)
    np.save("spectrum", spectrum)
