# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import logging
from typing import List, Union

from renormalizer.model import MolList, EphTable
from renormalizer.mps.backend import np, xp, USE_GPU
from renormalizer.mps import svd_qn
from renormalizer.mps.matrix import (
    asnumpy,
    asxp,
    allclose,
    backend,
    vstack,
    dstack,
    concatenate,
    zeros,
    tensordot,
    multi_tensor_contract,
    Matrix)
from renormalizer.utils import sizeof_fmt, CompressConfig
from renormalizer.mps.lib import (
    Environ,
    select_basis,
    )

import opt_einsum as oe

logger = logging.getLogger(__name__)


class MatrixProduct:

    def __init__(self):
        # XXX: when modify theses codes, keep in mind to update `metacopy` method
        # set to a list of None upon metacopy
        self._mp: List[Union[Matrix, None]] = []
        self.dtype = backend.real_dtype

        # in mpo.quasi_boson, mol_list is not set, then _ephtable and _pbond_list should be used
        self.mol_list: MolList = None
        self._ephtable: EphTable = None
        self._pbond_list: List[int] = None

        # mpo also need to be compressed sometimes
        self.compress_config: CompressConfig = CompressConfig()

        # QN related
        self.use_dummy_qn: bool = False
        # self.use_dummy_qn = True
        self.qn: List[List[int]] = []
        self.qnidx: int = None
        self._qntot: int = None
        # if sweeping to right: True else False
        self.to_right: bool = None

        # compress after add?
        self.compress_add: bool = False

    @property
    def site_num(self):
        return len(self._mp)

    @property
    def mol_num(self):
        return self.mol_list.mol_num

    @property
    def threshold(self):
        return self.compress_config.threshold

    @threshold.setter
    def threshold(self, v):
        self.compress_config.threshold = v

    @property
    def is_mps(self):
        raise NotImplementedError

    @property
    def is_mpo(self):
        raise NotImplementedError

    @property
    def is_mpdm(self):
        raise NotImplementedError

    @property
    def is_complex(self):
        return self.dtype == backend.complex_dtype

    @property
    def bond_dims(self) -> List:
        bond_dims = (
            [mt.bond_dim[0] for mt in self] + [self[-1].bond_dim[-1]]
            if self.site_num
            else []
        )
        # return a list so that the logging result is more pretty
        return bond_dims

    @property
    def bond_dims_mean(self) -> int:
        return int(round(np.mean(self.bond_dims)))

    @property
    def ephtable(self):
        if self._ephtable is not None:
            return self._ephtable
        else:
            return self.mol_list.ephtable

    @ephtable.setter
    def ephtable(self, ephtable: EphTable):
        self._ephtable = ephtable

    @property
    def pbond_list(self):
        if self._pbond_list is None:
            self._pbond_list = self.mol_list.pbond_list
        return self._pbond_list

    @pbond_list.setter
    def pbond_list(self, pbond_list):
        self._pbond_list = pbond_list

    @property
    def qntot(self):
        if self.use_dummy_qn:
            return 0
        else:
            return self._qntot

    @qntot.setter
    def qntot(self, qntot: int):
        if not self.use_dummy_qn:
            self._qntot = qntot

    def build_empty_qn(self):
        self.qntot = 0
        # set qnidx to the right to be consistent with most MPS/MPO setups
        if self.qnidx is None:
            self.qnidx = len(self) - 1
        self.qn = [[0] * dim for dim in self.bond_dims]
        if self.to_right is None:
            self.to_right = False

    def build_none_qn(self):
        self.qntot = None
        self.qnidx = None
        self.qn = None
        self.to_right = None

    def clear_qn(self):
        self.qntot = 0
        self.qn = [[0] * dim for dim in self.bond_dims]

    def move_qnidx(self, dstidx: int):
        """
        Quantum number has a boundary site, left hand of the site is L system qn,
        right hand of the side is R system qn, the sum of quantum number of L system
        and R system is tot.
        """
        # construct the L system qn
        for idx in range(self.qnidx + 1, self.site_num):
            self.qn[idx] = [self.qntot - i for i in self.qn[idx]]

        # set boundary to fsite:
        for idx in range(self.site_num - 1, dstidx, -1):
            self.qn[idx] = [self.qntot - i for i in self.qn[idx]]
        self.qnidx = dstidx

    def check_left_canonical(self, rtol=1e-5, atol=1e-8):
        """
        check L-canonical
        """
        for mt in self[:-1]:
            if not mt.check_lortho(rtol, atol):
                return False
        return True

    def check_right_canonical(self, rtol=1e-5, atol=1e-8):
        """
        check R-canonical
        """
        for mt in self[1:]:
            if not mt.check_rortho(rtol, atol):
                return False
        return True

    @property
    def is_left_canon(self):
        """
        check the qn center in the L-canonical structure
        """
        return self.qnidx == self.site_num - 1

    @property
    def is_right_canon(self):
        """
        check the qn center in the R-canonical structure
        """
        return self.qnidx == 0

    def ensure_left_canon(self, rtol=1e-5, atol=1e-8):
        if not self.check_left_canonical(rtol, atol):
            self.move_qnidx(0)
            self.to_right = True
            self.canonicalise()

    def ensure_right_canon(self, rtol=1e-5, atol=1e-8):
        if not self.check_right_canonical(rtol, atol):
            self.move_qnidx(self.site_num - 1)
            self.to_right = False
            self.canonicalise()

    def iter_idx_list(self, full: bool, stop_idx: int=None):
        # if not `full`, the last site is omitted.
        if self.to_right:
            if stop_idx is not None:
                last = stop_idx
            else:
                last = self.site_num if full else self.site_num - 1
            return range(self.qnidx, last)
        else:
            if stop_idx is not None:
                last = stop_idx
            else:
                last = -1 if full else 0
            return range(self.qnidx, last, -1)

    def _update_ms(
        self, idx: int, u: np.ndarray, vt: np.ndarray, sigma=None, qnlset=None, qnrset=None, m_trunc=None
    ):
        r""" update mps directly after svd
        
        """
        
        if m_trunc is None:
            m_trunc = u.shape[1]
        u = u[:, :m_trunc]
        vt = vt[:m_trunc, :]
        if sigma is None:
            # canonicalise, vt is not unitary
            if self.is_mpo:
                if self.to_right:
                    norm = np.linalg.norm(vt)
                    u *= norm
                    vt /= norm
                else:
                    norm = np.linalg.norm(u)
                    u /= norm
                    vt *= norm
        else:
            sigma = sigma[:m_trunc]
            if (not self.is_mpo and self.to_right) or (self.is_mpo and not self.to_right):
                vt = np.einsum("i, ij -> ij", sigma, vt)
            else:
                u = np.einsum("ji, i -> ji", u, sigma)
        if self.to_right:
            self[idx + 1] = tensordot(vt, self[idx + 1], axes=1)
            ret_mpsi = u.reshape(
                [u.shape[0] // self[idx].pdim_prod] + list(self[idx].pdim) + [m_trunc]
            )
            if qnlset is not None:
                self.qn[idx + 1] = qnlset[:m_trunc]
                self.qnidx = idx + 1
        else:
            self[idx - 1] = tensordot(self[idx - 1], u, axes=1)
            ret_mpsi = vt.reshape(
                [m_trunc] + list(self[idx].pdim) + [vt.shape[1] // self[idx].pdim_prod]
            )
            if qnrset is not None:
                self.qn[idx] = qnrset[:m_trunc]
                self.qnidx = idx - 1
        if ret_mpsi.nbytes < ret_mpsi.base.nbytes * 0.8:
            # do copy here to discard unnecessary data. Note that in NumPy common slicing returns
            # a `view` containing the original data. If `ret_mpsi` is used directly the original
            # `u` or `vt` is not garbage collected.
            ret_mpsi = ret_mpsi.copy()
        assert ret_mpsi.any()
        self[idx] = ret_mpsi

    def _switch_direction(self):
        assert self.to_right is not None
        if self.to_right:
            self.qnidx = self.site_num - 1
            self.to_right = False
            # assert self.check_left_canonical()
        else:
            self.qnidx = 0
            self.to_right = True
            # assert self.check_right_canonical()

    def _get_big_qn(self, cidx: List[int]):
        r""" get the quantum number of L-block and R-block renormalized basis 
        
        Parameters
        ----------
        cidx : list
            a list of center(active) site index. For 1site/2site algorithm, cidx
            has one/two elements.

        Returns
        -------
        qnbigl : np.ndarray
            super-L-block (L-block + active site if necessary) quantum number
        qnbigr : np.ndarray
            super-R-block (active site + R-block if necessary) quantum number
        qnmat : np.ndarray
            L-block + active site + R-block quantum number
        
        """

        if len(cidx) == 2:
            cidx = sorted(cidx)
            assert cidx[0]+1 == cidx[1]
        elif len(cidx) > 2:
            assert False
        assert self.qnidx in cidx
        
        sigmaqn = [np.array(self[idx].sigmaqn) for idx in cidx]
        qnl = np.array(self.qn[cidx[0]])
        qnr = np.array(self.qn[cidx[-1]+1])
        if len(cidx) == 1:
            if self.to_right:
                qnbigl = np.add.outer(qnl, sigmaqn[0])
                qnbigr = qnr
            else:
                qnbigl = qnl
                qnbigr = np.add.outer(sigmaqn[0], qnr)
        else:
            qnbigl = np.add.outer(qnl, sigmaqn[0])
            qnbigr = np.add.outer(sigmaqn[1], qnr)
        qnmat = np.add.outer(qnbigl, qnbigr)
        return qnbigl, qnbigr, qnmat

    def add(self, other: "MatrixProduct"):
        assert self.qntot == other.qntot
        assert self.site_num == other.site_num
        #assert self.qnidx == other.qnidx

        new_mps = self.metacopy()
        if other.dtype == backend.complex_dtype:
            new_mps.dtype = backend.complex_dtype
        if self.is_complex:
            new_mps.to_complex(inplace=True)
        new_mps.compress_config.update(self.compress_config)

        if self.is_mps:  # MPS
            new_mps[0] = dstack([self[0], other[0]])
            for i in range(1, self.site_num - 1):
                mta = self[i]
                mtb = other[i]
                pdim = mta.shape[1]
                assert pdim == mtb.shape[1]
                new_ms = zeros(
                    [mta.shape[0] + mtb.shape[0], pdim, mta.shape[2] + mtb.shape[2]],
                    dtype=new_mps.dtype,
                )
                new_ms[: mta.shape[0], :, : mta.shape[2]] = mta
                new_ms[mta.shape[0] :, :, mta.shape[2] :] = mtb
                new_mps[i] = new_ms

            new_mps[-1] = vstack([self[-1], other[-1]])
        elif self.is_mpo or self.is_mpdm:  # MPO
            new_mps[0] = concatenate((self[0], other[0]), axis=3)
            for i in range(1, self.site_num - 1):
                mta = self[i]
                mtb = other[i]
                pdimu = mta.shape[1]
                pdimd = mta.shape[2]
                assert pdimu == mtb.shape[1]
                assert pdimd == mtb.shape[2]

                new_ms = zeros(
                    [
                        mta.shape[0] + mtb.shape[0],
                        pdimu,
                        pdimd,
                        mta.shape[3] + mtb.shape[3],
                    ],
                    dtype=new_mps.dtype,
                )
                new_ms[: mta.shape[0], :, :, : mta.shape[3]] = mta[:, :, :, :]
                new_ms[mta.shape[0] :, :, :, mta.shape[3] :] = mtb[:, :, :, :]
                new_mps[i] = new_ms

            new_mps[-1] = concatenate((self[-1], other[-1]), axis=0)
        else:
            assert False

        new_mps.move_qnidx(other.qnidx)
        new_mps.to_right = other.to_right
        new_mps.qn = [qn1 + qn2 for qn1, qn2 in zip(self.qn, other.qn)]
        new_mps.qn[0] = [0]
        new_mps.qn[-1] = [0]
        if self.compress_add:
            new_mps.canonicalise()
            new_mps.compress()
        return new_mps

    def compress(self, temp_m_trunc=None, ret_s=False):
        """
        inp: canonicalise MPS (or MPO)

        side='l': compress LEFT-canonicalised MPS
                  by sweeping from RIGHT to LEFT
                  output MPS is right canonicalised i.e. CRRR

        side='r': reverse of 'l'

        Returns
        -------
             truncated MPS
        """
        if self.to_right:
            assert self.qnidx == 0
        else:
            assert self.qnidx == self.site_num-1
        
        if self.compress_config.bonddim_should_set:
            self.compress_config.set_bonddim(len(self)+1)
        # used for logging at exit
        sz_before = self.total_bytes
        if not self.is_mpo:
            # ensure mps is canonicalised. This is time consuming.
            # to disable this, run python as `python -O`
            if self.is_left_canon:
                assert self.check_left_canonical()
            else:
                assert self.check_right_canonical()
        system = "L" if self.to_right else "R"

        s_list = []
        for idx in self.iter_idx_list(full=False):
            mt: Matrix = self[idx]
            if self.to_right:
                mt = mt.l_combine()
            else:
                mt = mt.r_combine()
            qnbigl, qnbigr, _ = self._get_big_qn([idx])
            u, sigma, qnlset, v, sigma, qnrset = svd_qn.Csvd(
                mt.array,
                qnbigl,
                qnbigr,
                self.qntot,
                system=system,
                full_matrices=False,
            )
            vt = v.T
            s_list.append(sigma)
            if temp_m_trunc is None:
                m_trunc = self.compress_config.compute_m_trunc(
                    sigma, idx, self.to_right
                )
            else:
                m_trunc = min(temp_m_trunc, len(sigma))
            self._update_ms(
                idx, u, vt, sigma, qnlset, qnrset, m_trunc
            )

        self._switch_direction()
        compress_ratio = sz_before / self.total_bytes
        logger.debug(f"size before/after compress: {sizeof_fmt(sz_before)}/{sizeof_fmt(self.total_bytes)}, ratio: {compress_ratio}")
        if not ret_s:
            # usual exit
            return self
        else:
            # return singular value list
            return self, s_list
    
    def variational_compress(self, mpo=None, guess=None):
        r"""Variational compress an mps/mpdm/mpo
        
        Parameters
        ----------
        mpo : renormalizer.mps.Mpo, optional 
            Default is ``None``. if mpo is not ``None``, the returned mps is
            an approximation of ``mpo @ self``
        guess : renormalizer.mps.MatrixProduct, optional
            Initial guess of compressed mps/mpdm/mpo. Default is ``None``. 
        
        Note
        ----
        the variational compress related configurations is defined in
        ``self`` if ``guess=None``, otherwise is defined in ``guess``

        Returns
        -------
        mp : renormalizer.mps.MatrixProduct
            a new compressed mps/mpdm/mpo, ``self`` is not overwritten.
            ``guess`` is overwritten.
        
        """

        if mpo is None:
            logger.info("Recommend to use svd to compress a single mps/mpo/mpdm.")
            raise NotImplementedError
        
        if guess is None:
            # a minimal representation of self and mpo
            compressed_mpo = mpo.copy().canonicalise().compress(
                    temp_m_trunc=self.compress_config.vguess_m[0])
            compressed_mps = self.copy().canonicalise().compress(
                    temp_m_trunc=self.compress_config.vguess_m[1])
            # the attributes of guess would be the same as self
            guess = compressed_mpo.apply(compressed_mps)
        mps = guess
        mps.ensure_left_canon()
        logger.info(f"initial guess bond dims: {mps.bond_dims}")

        procedure = mps.compress_config.vprocedure
        method = mps.compress_config.vmethod

        environ = Environ(self, mpo, "L", mps_conj=mps.conj())
        
        converged = False
        for isweep, (mmax, percent) in enumerate(procedure):
            logger.debug(f"isweep: {isweep}")
            logger.debug(f"mmax, percent: {mmax}, {percent}")
            logger.debug(f"mps bond dims: {mps.bond_dims}")
            
            for imps in mps.iter_idx_list(full=True):
                if method == "2site" and \
                    ((mps.to_right and imps == mps.site_num-1)
                    or ((not mps.to_right) and imps == 0)):
                    break
                
                if mps.to_right:
                    lmethod, rmethod = "System", "Enviro"
                else:
                    lmethod, rmethod = "Enviro", "System"

                if method == "1site":
                    lidx = imps - 1
                    cidx= [imps]
                    ridx = imps + 1
                elif method == "2site":
                    if mps.to_right:
                        lidx = imps - 1
                        cidx = [imps, imps+1] 
                        ridx = imps + 2
                    else:
                        lidx = imps - 2
                        cidx = [imps-1, imps]  # center site
                        ridx = imps + 1
                else:
                    assert False
                logger.debug(f"optimize site: {cidx}")

                ltensor = environ.GetLR(
                    "L", lidx, self, mpo, itensor=None, method=lmethod, 
                    mps_conj = mps.conj()
                )
                rtensor = environ.GetLR(
                    "R", ridx, self, mpo, itensor=None, method=rmethod,
                    mps_conj = mps.conj()
                )

                # get the quantum number pattern
                qnbigl, qnbigr, qnmat = mps._get_big_qn(cidx)
                
                # center mo
                cmo = [asxp(mpo[idx]) for idx in cidx] 
                cms = [asxp(self[idx]) for idx in cidx]
                if method == "1site":
                    if cms[0].ndim == 3:
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
                    elif cms[0].ndim == 4:
                        # S-a   l-S
                        #     d
                        # O-b-O-f-O
                        #     e
                        # S-c   k-S
                        #     g
                        path = [
                            ([0, 2], "abc, bdef -> acdef"),
                            ([2, 0], "acdef, cegk -> adfgk"),
                            ([1, 0], "adfgk, lfk -> adgl"),
                        ]
                    cout = multi_tensor_contract(
                        path, ltensor, cms[0], cmo[0], rtensor
                    )
                else:
                    if USE_GPU:
                        oe_backend = "cupy"
                    else:
                        oe_backend = "numpy"
                    if cms[0].ndim == 3:
                        # S-a       l-S
                        #     d   g
                        # O-b-O-f-O-j-O
                        #     e   h
                        # S-c   m   k-S
                    
                        cout = oe.contract("abc, bdef, fghj, ljk, cem, mhk -> adgl",
                                ltensor, cmo[0], cmo[1], rtensor, cms[0], cms[1],
                                backend=oe_backend)
                    elif cms[0].ndim == 4:
                        # S-a       l-S
                        #     d   g
                        # O-b-O-f-O-j-O
                        #     e   h
                        # S-c   m   k-S
                        #     n   p
                        cout = oe.contract("abc, bdef, fghj, ljk, cenm, mhpk -> adngpl",
                                ltensor, cmo[0], cmo[1], rtensor, cms[0], cms[1],
                                backend=oe_backend)
                # clean up the elements which do not meet the qn requirements
                cout[qnmat!=mps.qntot] = 0
                mps._update_mps(cout, cidx, qnbigl, qnbigr, mmax, percent)
            
            mps._switch_direction()
            
            # check if convergence
            if isweep > 0 and percent == 0 and \
                    mps.distance(mps_old) / np.sqrt(mps.dot(mps.conj()).real) < mps.compress_config.vrtol:
                converged = True
                break
            
            mps_old = mps.copy()

        if converged:
            logger.info("Variational compress is converged!")
        else:
            logger.warning("Variational compress is not converged! Please increase the procedure!")
        
        # remove the redundant bond dimension near the boundary of the MPS
        mps.canonicalise()
        logger.info(f"{mps}")
        
        return mps
    
    def _update_mps(self, cstruct, cidx, qnbigl, qnbigr, Mmax, percent=0):
        r"""update mps with basis selection algorithm of J. Chem. Phys. 120,
        3172 (2004).
        
        Parameters
        ---------
        cstruct : ndarray
            The active site coefficient.
        cidx : list
            The List of active site index.
        qnbigl : ndarray
            The super-L-block quantum number.
        qnbigr : ndarray
            The super-R-block quantum number.
        Mmax : int
            The maximal bond dimension.
        percent : float, int
            The percentage of renormalized basis which is equally selected from
            each quantum number section rather than according to singular
            values. ``percent`` is defined in ``procedure`` of 
            `renormalizer.utils.configs.OptimizeConfig` and ``vprocedure`` of
            `renormalizer.utils.configs.CompressConfig`.

        Returns
        -------
        None : 
            ``self`` is overwritten inplace.

        """
        
        system = "L" if self.to_right else "R"

        # step 1: get the selected U, S, V
        if type(cstruct) is not list:
            # SVD method
            # full_matrices = True here to enable increase the bond dimension
            Uset, SUset, qnlnew, Vset, SVset, qnrnew = svd_qn.Csvd(
                asnumpy(cstruct), qnbigl, qnbigr, self.qntot, system=system
            )

            if self.to_right:
                ms, msdim, msqn, compms = select_basis(
                    Uset, SUset, qnlnew, Vset, Mmax, percent=percent
                )
                ms = ms.reshape(list(qnbigl.shape) + [msdim])
                compms = xp.moveaxis(compms.reshape(list(qnbigr.shape) + [msdim]), -1, 0)

            else:
                ms, msdim, msqn, compms = select_basis(
                    Vset, SVset, qnrnew, Uset, Mmax, percent=percent
                )
                ms = xp.moveaxis(ms.reshape(list(qnbigr.shape) + [msdim]), -1, 0) 
                compms = compms.reshape(list(qnbigl.shape) + [msdim])
        
        else:
            # state-averaged method
            ddm = 0.0
            for iroot in range(len(cstruct)):
                if self.to_right:
                    ddm += tensordot(
                        cstruct[iroot],
                        cstruct[iroot],
                        axes=(
                            range(qnbigl.ndim, cstruct[iroot].ndim),
                            range(qnbigl.ndim, cstruct[iroot].ndim),
                        ),
                    )
                else:
                    ddm += tensordot(
                        cstruct[iroot],
                        cstruct[iroot],
                        axes=(range(qnbigl.ndim), range(qnbigl.ndim)),
                    )
            ddm /= len(cstruct)
            Uset, Sset, qnnew = svd_qn.Csvd(asnumpy(ddm), qnbigl, qnbigr, self.qntot,
                    system=system, ddm=True)
            ms, msdim, msqn, compms = select_basis(
                Uset, Sset, qnnew, None, Mmax, percent=percent
            )

            if self.to_right:
                ms = ms.reshape(list(qnbigl.shape) + [msdim])
                compms = tensordot(
                        ms.reshape(list(qnbigl.shape) + [msdim]),
                        cstruct[0],
                        axes=(range(qnbigl.ndim), range(qnbigl.ndim)),
                        )
            else:
                ms = xp.moveaxis(ms.reshape(list(qnbigr.shape) + [msdim]), -1, 0)
                compms = tensordot(
                        cstruct[0],
                        ms.reshape(list(qnbigr.shape) + [msdim]),
                        axes=(range(qnbigl.ndim, cstruct[0].ndim), range(qnbigr.ndim)),
                        )
        # step 2, put updated U, S, V back to self
        if len(cidx) == 1:
            # 1site method
            self[cidx[0]] = ms
            if self.to_right:
                if cidx[0] != self.site_num - 1:
                    self[cidx[0] + 1] = tensordot(compms, self[cidx[0] + 1], axes=1)
                    self.qn[cidx[0] + 1] = msqn
                    self.qnidx = cidx[0] + 1
                else:
                    self[cidx[0]] = tensordot(self[cidx[0]], compms, axes=1)
                    self.qnidx = self.site_num - 1
            else:
                if cidx[0] != 0:
                    self[cidx[0] - 1] = tensordot(self[cidx[0] - 1], compms, axes=1)
                    self.qn[cidx[0]] = msqn
                    self.qnidx = cidx[0] - 1
                else:
                    self[cidx[0]] = tensordot(compms, self[cidx[0]], axes=1)
                    self.qnidx = 0
        else:
            if self.to_right:
                self[cidx[0]] = ms
                self[cidx[1]] = compms
                self.qnidx = cidx[1]
            else:
                self[cidx[1]] = ms
                self[cidx[0]] = compms
                self.qnidx = cidx[0]

            self.qn[cidx[1]] = msqn


    def canonicalise(self, stop_idx: int=None, normalize=False):
        # stop_idx: mix canonical site at `stop_idx`
        if self.to_right:
            assert self.qnidx == 0
        else:
            assert self.qnidx == self.site_num-1

        for idx in self.iter_idx_list(full=False, stop_idx=stop_idx):
            mt: Matrix = self[idx]
            assert mt.any()
            if self.to_right:
                mt = mt.l_combine()
            else:
                mt = mt.r_combine()
            qnbigl, qnbigr, _ = self._get_big_qn([idx])
            system = "L" if self.to_right else "R"
            u, qnlset, v, qnrset = svd_qn.Csvd(
                mt.array,
                qnbigl,
                qnbigr,
                self.qntot,
                QR=True,
                system=system,
                full_matrices=False,
            )
            if normalize:
                # roughly normalize. Used when the each site of the mps is scaled such as in exact thermal prop
                v /= np.linalg.norm(v[:, 0])
            self._update_ms(
                idx, u, v.T, sigma=None, qnlset=qnlset, qnrset=qnrset
            )
        # can't iter to idx == 0 or idx == self.site_num - 1
        if (not self.to_right and idx == 1) or (self.to_right and idx == self.site_num - 2):
            self._switch_direction()
        return self

    def conj(self):
        """
        complex conjugate
        """
        new_mp = self.metacopy()
        for idx, mt in enumerate(self):
            new_mp[idx] = mt.conj()
        return new_mp

    def dot(self, other: "MatrixProduct") -> complex:
        """
        dot product of two mps / mpo
        """

        assert len(self) == len(other)
        e0 = xp.eye(1, 1)
        # for debugging. It has little computational cost anyway
        debug_t = []
        for mt1, mt2 in zip(self, other):
            # sum_x e0[:,x].m[x,:,:]
            debug_t.append(e0)
            e0 = tensordot(e0, mt2.array, 1)
            # sum_ij e0[i,p,:] self[i,p,:]
            # note, need to flip a (:) index onto top,
            # therefore take transpose
            if mt1.ndim == 3:
                e0 = tensordot(e0, mt1.array, ([0, 1], [0, 1])).T
            elif mt1.ndim == 4:
                e0 = tensordot(e0, mt1.array, ([0, 1, 2], [0, 1, 2])).T
            else:
                assert False

        return complex(e0[0, 0])

    def angle(self, other):
        return abs(self.conj().dot(other))

    def scale(self, val, inplace=False):
        new_mp = self if inplace else self.copy()
        # np.iscomplex regards 1+0j as non complex while np.iscomplexobj
        # regards 1+0j as complex. The former is the desired behavior
        if np.iscomplex(val):
            new_mp.to_complex(inplace=True)
        else:
            val = val.real
        assert new_mp[self.qnidx].array.any()
        new_mp[self.qnidx] = new_mp[self.qnidx] * val
        return new_mp

    def to_complex(self, inplace=False):
        if inplace:
            new_mp = self
        else:
            new_mp = self.metacopy()
        new_mp.dtype = backend.complex_dtype
        for i, mt in enumerate(self):
            if mt is None:
                # dummy mt after metacopy. Bad idea. Remove the dummy thing when feasible
                continue
            new_mp[i] = mt.to_complex()
        return new_mp

    def distance(self, other) -> float:
        l1 = self.conj().dot(self)
        l2 = other.conj().dot(other)
        l1dotl2 = self.conj().dot(other)
        dis_square = (l1 + l2
            - l1dotl2
            - l1dotl2.conjugate()).real

        if dis_square < 0:
            assert dis_square/l1.real < 1e-8
            res = 0.
        else:
            res = np.sqrt(dis_square).item()

        return float(res)

    def copy(self):
        new = self.metacopy()
        new._mp = [m.copy() for m in self._mp]
        return new

    # only (shalow) copy metadata because usually after been copied the real data is overwritten
    def metacopy(self) -> "MatrixProduct":
        new = self.__class__.__new__(self.__class__)
        new._mp = [None] * len(self)
        new.dtype = self.dtype
        new.mol_list = self.mol_list
        new._ephtable = self._ephtable
        new._pbond_list = self._pbond_list
        # need to deep copy compress_config because threshold might change dynamically
        new.compress_config = self.compress_config.copy()
        new.peak_bytes = 0
        new.use_dummy_qn = self.use_dummy_qn
        new.qn = [qn.copy() for qn in self.qn]
        new.qnidx = self.qnidx
        new._qntot = self.qntot
        new.to_right = self.to_right
        new.compress_add = self.compress_add
        return new

    def _array2mt(self, array, idx):
        if isinstance(array, Matrix):
            mt = array.astype(self.dtype)
        else:
            mt = Matrix(array, dtype=self.dtype)
        if self.use_dummy_qn:
            mt.sigmaqn = np.zeros(mt.pdim_prod, dtype=np.int)
        else:
            mt.sigmaqn = self._get_sigmaqn(idx)
        return mt
    
    def build_empty_mp(self, num):
        self._mp = [[None]] * num

    @property
    def total_bytes(self):
        return sum(array.nbytes for array in self)

    def _get_sigmaqn(self, idx):
        raise NotImplementedError

    def get_sigmaqn(self, idx):
        return self[idx].sigmaqn

    def set_threshold(self, val):
        self.compress_config.threshold = val

    def __eq__(self, other):
        for m1, m2 in zip(self, other):
            if not allclose(m1, m2):
                return False
        return True

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "%s with %d sites" % (self.__class__, len(self))

    def __iter__(self):
        return iter(self._mp)

    def __len__(self):
        # The same semantic with `list`
        return len(self._mp)

    def __mul__(self, other):
        assert isinstance(other, (float, complex))
        return self.scale(other)

    def __rmul__(self, other):
        assert isinstance(other, (float, complex))
        return self.scale(other)

    def __getitem__(self, item):
        return self._mp[item]

    def __setitem__(self, key, array):
        new_mt = self._array2mt(array, key)
        self._mp[key] = new_mt

    def __add__(self, other: "MatrixProduct"):
        return self.add(other)

    def __sub__(self, other: "MatrixProduct"):
        return self.add(other.scale(-1))

    def append(self, array):
        new_mt = self._array2mt(array, len(self))
        self._mp.append(new_mt)
    
    def __str__(self):
        template_str = "current size: {}, Matrix product bond dim:{}"
        return template_str.format(sizeof_fmt(self.total_bytes), self.bond_dims,)
