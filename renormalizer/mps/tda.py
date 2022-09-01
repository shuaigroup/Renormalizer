# -*- coding: utf-8 -*-

import logging
from collections import defaultdict

import numpy as np
import scipy
import opt_einsum as oe

from renormalizer.mps.matrix import tensordot, multi_tensor_contract, asnumpy, asxp
from renormalizer.mps.backend import xp, USE_GPU, primme, IMPORT_PRIMME_EXCEPTION
from renormalizer.mps import Mps
from renormalizer.mps.lib import Environ, compressed_sum
from renormalizer.lib import davidson

logger = logging.getLogger(__name__)

class TDA(object):
    r""" Tamm–Dancoff approximation (or called CIS) to calculate the excited
    states based on MPS. TDA use the first order tangent space to do excitation.
    The implementation is similar to J. Chem. Phys. 140, 024108 (2014).
    
    Parameters
    ----------
    model: renormalizer.model.model
        Model of the system
    hmpo: renormalizer.mps.Mpo
        mpo of Hamiltonian
    mps: renormalizer.mps.Mps
        ground state mps (will be overwritten)
    nroots: int, optional
        number of roots to be calculated. Default is ``1``.
    algo: str, optional
        iterative diagonalization solver. Default is ``"primme"`` if ``primme`` is installed.
        Valid options are ``davidson`` and ``primme``.

    Note
    ----
    Quantum number is not used, thus the conservation is not guaranteed.

    """

    def __init__(self, model, hmpo, mps, nroots=1, algo=None):
               
        self.model = model
        self.hmpo = hmpo
        self.mps = mps  # mps will be overwritten inplace
        self.nroots = nroots
        if algo is None:
            if primme is not None:
                self.algo = "primme"
            else:
                self.algo = "davidson"
        else:
            self.algo = algo

        self.e = None
        # wavefunction: [mps_l_cano, mps_r_cano, tangent_u, tda_coeff_list]
        self.wfn = None
        self.configs = defaultdict(list)

    def kernel(self, restart=False, include_psi0=False):
        r"""calculate the roots

        Parameters
        ----------
        restart: bool, optional
            if restart from the former converged root. Default is ``False``.
            If ``restart = True``, ``include_psi0`` must be the same as the
            former calculation.
        include_psi0: bool, optional
            if the basis of Hamiltonian includes the ground state
                :math:`\Psi_0`. Default is ``False``.

        Returns
        -------
        e: np.ndarray
            the energy of the states, if ``include_psi0 = True``, the first
            element is the ground state energy, otherwise, it is the energy of
            the first excited state.

        """
        # right canonical mps
        mpo = self.hmpo
        nroots = self.nroots
        algo = self.algo
        site_num = mpo.site_num

        if not restart:
            # make sure that M is not redundant near the edge
            mps = self.mps.ensure_right_canonical().canonicalise().normalize("mps_and_coeff").canonicalise()
            logger.debug(f"reference mps shape, {mps}")
            mps_r_cano = mps.copy()
            assert mps.to_right 
            
            tangent_u = []
    
            for ims in range(len(mps)):
                
                shape = list(mps[ims].shape)
                u, s, vt = scipy.linalg.svd(mps[ims].l_combine(), full_matrices=True)
                rank = len(s)
                if include_psi0 and ims == site_num-1: 
                    tangent_u.append(u.reshape(shape[:-1]+[-1]))
                else:
                    if rank < u.shape[1]:
                        tangent_u.append(u[:,rank:].reshape(shape[:-1]+[-1]))
                    else:
                        tangent_u.append(None)  # the tangent space is None

                mps[ims] = u[:,:rank].reshape(shape[:-1]+[-1])
                
                vt = xp.einsum("i, ij -> ij", asxp(s), asxp(vt))
                if ims == site_num-1:
                    assert vt.size == 1 and xp.allclose(vt, 1)
                else:
                    mps[ims+1] = asnumpy(tensordot(vt, mps[ims+1], ([-1],[0])))
                
            mps_l_cano = mps.copy() 
            mps_l_cano.to_right = False
            mps_l_cano.qnidx = site_num-1

        else:
            mps_l_cano, mps_r_cano, tangent_u, tda_coeff_list = self.wfn
            cguess = []
            for iroot in range(len(tda_coeff_list)):
                tda_coeff = tda_coeff_list[iroot]
                x = [c.flatten() for c in tda_coeff if c is not None]
                x = np.concatenate(x,axis=None)
                cguess.append(x)
            cguess = np.stack(cguess, axis=1)

        xshape = [] 
        xsize = 0
        for ims in range(site_num):
            if tangent_u[ims] is None:
                xshape.append((0,0))
            else:
                if ims == site_num-1:
                    xshape.append((tangent_u[ims].shape[-1], 1))
                else:    
                    xshape.append((tangent_u[ims].shape[-1], mps_r_cano[ims+1].shape[0]))
                xsize += np.prod(xshape[-1])
        
        logger.debug(f"DMRG-TDA H dimension: {xsize}")
        
        if USE_GPU:
            oe_backend = "cupy"
        else:
            oe_backend = "numpy"
        
        mps_tangent = mps_r_cano.copy()
        environ = Environ(mps_tangent, mpo, "R")
        hdiag = []
        for ims in range(site_num):
            ltensor = environ.GetLR(
                "L", ims-1, mps_tangent, mpo, itensor=None,
                method="System"
            )
            rtensor = environ.GetLR(
                "R", ims+1, mps_tangent, mpo, itensor=None,
                method="Enviro"
            )
            if tangent_u[ims] is not None:
                u = asxp(tangent_u[ims])
                tmp = oe.contract("abc, ded, bghe, agl, chl -> ld", ltensor, rtensor,
                        asxp(mpo[ims]), u, u, backend=oe_backend)   
                hdiag.append(asnumpy(tmp))
            mps_tangent[ims] = mps_l_cano[ims]
        hdiag = np.concatenate(hdiag, axis=None)
    
        count = 0
        
        # recover the vector-like x back to the ndarray tda_coeff
        def reshape_x(x):
            tda_coeff = []
            offset = 0
            for shape in xshape:
                if shape == (0,0):
                    tda_coeff.append(None)
                else:
                    size = np.prod(shape)
                    tda_coeff.append(x[offset:size+offset].reshape(shape))
                    offset += size
            
            assert offset == xsize
            return tda_coeff
            
        def hop(x):
            # H*X
            nonlocal count
            count += 1
            
            assert len(x) == xsize
            tda_coeff = reshape_x(x)
    
            res = [np.zeros_like(coeff) if coeff is not None else None for coeff in tda_coeff]
            
            # fix ket and sweep bra and accumulate into res
            for ims in range(site_num):
                if tda_coeff[ims] is None:
                    assert tangent_u[ims] is None
                    continue
                
                # mix-canonical mps
                mps_tangent = merge(mps_l_cano, mps_r_cano, ims+1)
                mps_tangent[ims] = tensordot(tangent_u[ims], tda_coeff[ims], (-1, 0))
                
                mps_tangent_conj = mps_r_cano.copy()
                environ = Environ(mps_tangent, mpo, "R", mps_conj=mps_tangent_conj)
                
                for ims_conj in range(site_num):
                    ltensor = environ.GetLR(
                        "L", ims_conj-1, mps_tangent, mpo, itensor=None,
                        mps_conj=mps_tangent_conj,
                        method="System"
                    )
                    rtensor = environ.GetLR(
                        "R", ims_conj+1, mps_tangent, mpo, itensor=None,
                        mps_conj=mps_tangent_conj,
                        method="Enviro"
                    )
                    if tda_coeff[ims_conj] is not None:
                        # S-a   l-S
                        #     d
                        # O-b-O-f-O
                        #     e
                        # S-c   k-S
    
                        path = [
                            ([0, 1], "abc, cek -> abek"),
                            ([2, 0], "abek, bdef -> akdf"),
                            ([1, 0], "akdf, lfk -> adl"),
                        ]
                        out = multi_tensor_contract(
                            path, ltensor, asxp(mps_tangent[ims_conj]),
                            asxp(mpo[ims_conj]), rtensor
                        )
                        res[ims_conj] += asnumpy(tensordot(tangent_u[ims_conj], out,
                            ([0,1], [0,1])))
                    
                    # mps_conj combine 
                    mps_tangent_conj[ims_conj] = mps_l_cano[ims_conj]    
            
            res = [mat for mat in res if mat is not None]
    
            return np.concatenate(res, axis=None)
        
        if algo == "davidson":
            if restart:
                cguess = [cguess[:,i] for i in range(cguess.shape[1])]
            else:
                cguess = [np.random.random(xsize) - 0.5]
            precond = lambda x, e, *args: x / (hdiag - e + 1e-4)
            
            e, c = davidson(
                hop, cguess, precond, max_cycle=100,
                nroots=nroots, max_memory=64000
            )
            if nroots == 1:
                c = [c]
            c = np.stack(c, axis=1)

        elif algo == "primme":
            if primme is None:
                logger.error("can not import primme")
                raise IMPORT_PRIMME_EXCEPTION

            if not restart:
                cguess = None

            def multi_hop(x):
                if x.ndim == 1:
                    return hop(x)
                elif x.ndim == 2:
                    return np.stack([hop(x[:,i]) for i in range(x.shape[1])],axis=1)
                else:
                    assert False
    
            def precond(x): 
                if x.ndim == 1:
                    return np.einsum("i, i -> i", 1/(hdiag+1e-4), x)
                elif x.ndim ==2:
                    return np.einsum("i, ij -> ij", 1/(hdiag+1e-4), x)
                else:
                    assert False
            A = scipy.sparse.linalg.LinearOperator((xsize,xsize),
                    matvec=multi_hop, matmat=multi_hop)
            M = scipy.sparse.linalg.LinearOperator((xsize,xsize),
                    matvec=precond, matmat=precond)
            e, c = primme.eigsh(A, k=min(nroots,xsize), which="SA", 
                    v0=cguess,
                    OPinv=M,
                    method="PRIMME_DYNAMIC", 
                    tol=1e-6)
        else:
            assert False

        logger.debug(f"H*C times: {count}")
        
        tda_coeff_list = []
        for iroot in range(nroots):
            tda_coeff_list.append(reshape_x(c[:,iroot])) 
        
        self.e = np.array(e)
        self.wfn = [mps_l_cano, mps_r_cano, tangent_u, tda_coeff_list]
        return self.e

    def dump_wfn(self):
        r""" Dump wavefunction for restart and analysis
        
        Note
        ----
        mps_l_cano.npz: left-canonical form of initial mps
        mps_r_cano.npz: right-canonical form of the initial mps
        tangent_u: the tangent space u of the mixed-canonical mps
        tda_coeff_{iroot}.npz: the tda_coeff of the ith root.
        """
        
        mps_l_cano, mps_r_cano, tangent_u, tda_coeff_list = self.wfn
        
        # store mps_l_cano mps_r_cano
        mps_l_cano.dump("mps_l_cano.npz")
        mps_r_cano.dump("mps_r_cano.npz")
        
        # store tangent_u
        tangent_u_dict = {f"{i}":mat for i, mat in enumerate(tangent_u) if mat is
                not None}
        np.savez(f"tangent_u.npz", **tangent_u_dict)

        # store tda coeff
        for iroot, tda_coeff in enumerate(tda_coeff_list):
            tda_coeff_dict = {f"{i}":mat for i, mat in
                    enumerate(tda_coeff) if mat is not None}
            np.savez(f"tda_coeff_{iroot}.npz", **tda_coeff_dict)
 

    def load_wfn(self, model):
        r"""Load tda wavefunction
        """
        mps_l_cano = Mps.load(model, "mps_l_cano.npz")
        mps_r_cano = Mps.load(model, "mps_r_cano.npz")
        tangent_u_dict = np.load("tangent_u.npz")
        tangent_u = [tangent_u_dict[str(i)] if str(i) in tangent_u_dict.keys()
                else None for i in range(mps_l_cano.site_num)]
        tda_coeff_list = []
        for iroot in range(self.nroots):
            tda_coeff_dict = np.load(f"tda_coeff_{iroot}.npz")
            tda_coeff = [tda_coeff_dict[str(i)] if str(i) in tda_coeff_dict.keys()
                else None for i in range(mps_l_cano.site_num)]
            tda_coeff_list.append(tda_coeff)

        self.wfn = [mps_l_cano, mps_r_cano, tangent_u, tda_coeff_list]
 

    def analysis_1ordm(self):
        r""" calculate one-orbital reduced density matrix of each tda root
        """
        
        mps_l_cano, mps_r_cano, tangent_u, tda_coeff_list = self.wfn
        for iroot in range(self.nroots):
            tda_coeff = tda_coeff_list[iroot]
            rdm = None
            for ims in range(mps_l_cano.site_num):
                if tangent_u[ims] is None:
                    assert tda_coeff[ims] is None
                    continue
                mps_tangent = merge(mps_l_cano, mps_r_cano, ims+1) 
                mps_tangent[ims] = tensordot(tangent_u[ims], tda_coeff[ims],[-1,0]) 
                rdm_increment = mps_tangent.calc_1ordm()

                if rdm is None:
                    rdm = rdm_increment
                else:
                    rdm = [mat1+mat2 for mat1, mat2 in zip(rdm, rdm_increment)]
            
            dominant_config = {}
            for isite, mat in enumerate(rdm):
                quanta = np.argmax(np.diag(mat))
                dominant_config[isite] = (quanta, np.diag(mat)[quanta])
            logger.info(f"root: {iroot}, config: {dominant_config}")
            
    
    def analysis_dominant_config(self, thresh=0.8, alias=None, tda_m_trunc=20,
            return_compressed_mps=False):
        r""" analyze the dominant configuration of each tda root.
            The algorithm is to compress the tda wavefunction to a rank-1 Hartree
            state and get the ci coefficient of the largest configuration.
            Then, the configuration is subtracted from the tda wavefunction and
            redo the first step to get the second largest configuration. The
            two steps continue until the thresh is achieved.
        
        Parameters
        ----------
        thresh: float, optional
            the threshold to stop the analysis procedure of each root. 
            :math:`\sum_i |c_i|^2 > thresh`. Default is 0.8.
        alias: dict, optional
            The alias of each site. For example, ``alias={0:"v_0", 1:"v_2",
            2:"v_1"}``. Default is `None`. 
        tda_m_trunc: int, optional
            the ``m`` to compress a tda wavefunction. Default is 20.
        return_compressed_mps: bool, optional
            If ``True``, return the tda excited state as a single compressed
            mps. Default is `False`.
        
        Returns
        -------
        configs: dict
            The dominant configration of each root.
            ``configs = {0:[(config0, config_name0, ci_coeff0),(config1,
            config_name1, ci_coeff1),...], 1:...}``
        compressed_mps: List[renormalizer.mps.Mps]
            see the description in ``return_compressed_mps``.
        
        Note
        ----
        The compressed_mps is an approximation of the tda wavefunction with
        ``m=tda_m_trunc``.
        """

        mps_l_cano, mps_r_cano, tangent_u, tda_coeff_list = self.wfn
            
        if alias is not None:
            assert len(alias) == mps_l_cano.site_num
        
        compressed_mps = []
        for iroot in range(self.nroots):
            logger.info(f"iroot: {iroot}")
            tda_coeff = tda_coeff_list[iroot]
            mps_tangent_list = []
            weight = []
            for ims in range(mps_l_cano.site_num):
                if tangent_u[ims] is None:
                    assert tda_coeff[ims] is None
                    continue
                weight.append(np.sum(tda_coeff[ims]**2))
                mps_tangent = merge(mps_l_cano, mps_r_cano, ims+1) 
                mps_tangent[ims] = asnumpy(tensordot(tangent_u[ims],
                    tda_coeff[ims],[-1,0]))
                mps_tangent_list.append(mps_tangent)
            
            assert np.allclose(np.sum(weight), 1)
            # sort the mps_tangent from large weight to small weight
            mps_tangent_list = [mps_tangent_list[i] for i in np.argsort(weight,axis=None)[::-1]]

            coeff_square_sum = 0
            mps_delete = None
            
            config_visited = []
            while coeff_square_sum < thresh:
                if mps_delete is None:
                    # first compress it to M=tda_m_trunc
                    mps_rank1 = compressed_sum(mps_tangent_list, batchsize=5,
                            temp_m_trunc=tda_m_trunc)
                else:
                    mps_rank1 = compressed_sum([mps_delete] + mps_tangent_list,
                            batchsize=5, temp_m_trunc=tda_m_trunc)
                if coeff_square_sum == 0 and return_compressed_mps:
                    compressed_mps.append(mps_rank1.copy())       
                mps_rank1 = mps_rank1.canonicalise().compress(temp_m_trunc=1)
                
                # get config with the largest coeff
                config = []
                for ims, ms in enumerate(mps_rank1):
                    ms = ms.array.flatten()**2
                    quanta = int(np.argmax(ms))
                    config.append(quanta)
               
                # check if the config has been visited
                if config in config_visited:
                    break
                
                config_visited.append(config)

                ci_coeff_list = []
                for mps_tangent in mps_tangent_list:
                    sentinel = xp.ones((1,1))
                    for ims, ms in enumerate(mps_tangent):
                        sentinel = sentinel.dot(asxp(ms[:,config[ims],:]))
                    ci_coeff_list.append(float(sentinel[0,0]))
                ci_coeff = np.sum(ci_coeff_list)
                coeff_square_sum += ci_coeff**2
                
                if alias is not None:
                    config_name = [f"{quanta}"+f"{alias[isite]}" for isite, quanta
                            in enumerate(config) if quanta != 0]
                    config_name = " ".join(config_name)
                    self.configs[iroot].append((config, config_name, ci_coeff))
                    logger.info(f"config: {config}, {config_name}")
                else:
                    self.configs[iroot].append((config, ci_coeff))
                    logger.info(f"config: {config}")

                logger.info(f"ci_coeff: {ci_coeff}, weight:{ci_coeff**2}")

                condition = {dof:config[idof] for idof, dof in
                        enumerate(self.model.dofs)}
                mps_delete_increment = Mps.hartree_product_state(self.model, condition).scale(-ci_coeff)
                if mps_delete is None:
                    mps_delete = mps_delete_increment
                else:
                    mps_delete = mps_delete + mps_delete_increment

            logger.info(f"coeff_square_sum: {coeff_square_sum}")
        
        return self.configs, compressed_mps
        
def merge(mpsl, mpsr, idx):
    """ merge two mps (mpsl, mpsr) at dix
        idx belongs mpsr, the other attributes are the same aas mpsl
    """ 
    mps = mpsl.copy()
    for imps in range(idx, mpsr.site_num):
        mps[imps] = mpsr[imps]
    return mps
