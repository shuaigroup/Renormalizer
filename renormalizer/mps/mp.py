# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import logging
import os
import shutil
from typing import List, Union

from renormalizer.model import Model, HolsteinModel
from renormalizer.mps.backend import np, xp
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
    Matrix)
from renormalizer.mps.lib import (
    Environ,
    select_basis,
    )
from renormalizer.mps.hop_expr import hop_expr
from renormalizer.utils import sizeof_fmt, CompressConfig, OFS, calc_vn_entropy

logger = logging.getLogger(__name__)


class MatrixProduct:

    @classmethod
    def load(cls, model: Model, fname: str):
        npload = np.load(fname, allow_pickle=True)
        mp = cls()
        mp.model = model
        for i in range(int(npload["nsites"])):
            mt = npload[f"mt_{i}"]
            if np.iscomplexobj(mt):
                mp.dtype = backend.complex_dtype
            else:
                mp.dtype = backend.real_dtype
            mp.append(mt)
        mp.qn = npload["qn"]
        mp.qnidx = int(npload["qnidx"])
        mp.qntot = int(npload["qntot"])
        mp.to_right = bool(npload["to_right"])
        return mp

    def __init__(self):
        # XXX: when modify theses codes, keep in mind to update `metacopy` method
        # set to a list of None upon metacopy. String is used when the matrix is
        # stored in disks
        self._mp: List[Union[Matrix, None, str]] = []
        self.dtype = backend.real_dtype

        self.model: Model = None

        # mpo also need to be compressed sometimes
        self.compress_config: CompressConfig = CompressConfig()

        # QN related
        self.qn: List[List[int]] = []
        self.qnidx: int = None
        self.qntot: int = None
        # if sweeping to right: True else False
        self.to_right: bool = None

        # compress after add?
        self.compress_add: bool = False

    @property
    def site_num(self):
        return len(self._mp)

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


    vbond_list = vbond_dims = bond_list = bond_dims

    @property
    def bond_dims_mean(self) -> int:
        return int(round(np.mean(self.bond_dims)))

    @property
    def pbond_dims(self):
        return self.model.pbond_list

    pbond_list = pbond_dims

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

    def check_left_canonical(self, atol=None):
        """
        check L-canonical
        """
        for i in range(len(self)-1):
            if not self[i].check_lortho(atol):
                return False
        return True

    def check_right_canonical(self, atol=None):
        """
        check R-canonical
        """
        for i in range(1, len(self)):
            if not self[i].check_rortho(atol):
                return False
        return True

    @property
    def is_left_canonical(self):
        """
        check the qn center in the L-canonical structure
        """
        return self.qnidx == self.site_num - 1

    @property
    def is_right_canonical(self):
        """
        check the qn center in the R-canonical structure
        """
        return self.qnidx == 0

    def ensure_left_canonical(self, atol=None):
        if self.to_right or self.qnidx != self.site_num-1 or \
                (not self.check_left_canonical(atol)):
            self.move_qnidx(0)
            self.to_right = True
            return self.canonicalise()
        else:
            return self

    def ensure_right_canonical(self, atol=None):
        if (not self.to_right) or self.qnidx != 0 or \
                (not self.check_right_canonical(atol)):
            self.move_qnidx(self.site_num - 1)
            self.to_right = False
            return self.canonicalise()
        else:
            return self

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

    def _get_big_qn(self, cidx: List[int], swap=False):
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

        sigmaqn = [np.array(self._get_sigmaqn(idx)) for idx in cidx]
        if swap:
            assert len(sigmaqn) == 2
            sigmaqn = sigmaqn[::-1]
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

    @property
    def mp_norm(self) -> float:
        # the fast version in the comment rarely makes sense because in a lot of cases
        # the mps is not canonicalised (though qnidx is set)
        """
        if self.is_left_canon:
            assert self.check_left_canonical()
            return np.linalg.norm(np.ravel(self[-1]))
        else:
            assert self.check_right_canonical()
            return np.linalg.norm(np.ravel(self[0]))
        """
        res = self.conj().dot(self).real
        if res < 0:
            assert np.abs(res) < 1e-8
            res = 0
        res = np.sqrt(res)

        return float(res)

    def add(self, other: "MatrixProduct"):
        assert self.qntot == other.qntot
        assert self.site_num == other.site_num

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

        #assert self.qnidx == other.qnidx
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
            if self.is_left_canonical:
                assert self.check_left_canonical()
            else:
                assert self.check_right_canonical()
        system = "L" if self.to_right else "R"

        s_list = []
        for idx in self.iter_idx_list(full=False):
            mt: Matrix = self[idx]
            qnbigl, qnbigr, _ = self._get_big_qn([idx])
            u, sigma, qnlset, v, sigma, qnrset = svd_qn.svd_qn(
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
            raise NotImplementedError("Recommend to use svd to compress a single mps/mpo/mpdm.")

        if guess is None:
            # a minimal representation of self and mpo
            compressed_mpo = mpo.copy().canonicalise().compress(
                    temp_m_trunc=self.compress_config.vguess_m[0])
            compressed_mps = self.copy().canonicalise().compress(
                    temp_m_trunc=self.compress_config.vguess_m[1])
            # the attributes of guess would be the same as self
            guess = compressed_mpo.apply(compressed_mps)
        mps = guess
        mps.ensure_left_canonical()
        logger.info(f"initial guess bond dims: {mps.bond_dims}")

        procedure = mps.compress_config.vprocedure
        method = mps.compress_config.vmethod

        environ = Environ(self, mpo, "L", mps_conj=mps.conj())

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

                # todo: avoid the conjugations
                ltensor = environ.GetLR(
                    "L", lidx, self, mpo, itensor=None, method=lmethod,
                    mps_conj=mps.conj()
                )
                rtensor = environ.GetLR(
                    "R", ridx, self, mpo, itensor=None, method=rmethod,
                    mps_conj=mps.conj()
                )

                # get the quantum number pattern
                qnbigl, qnbigr, qnmat = mps._get_big_qn(cidx)

                # center mo
                cmo = [asxp(mpo[idx]) for idx in cidx]
                if method == "1site":
                    cms = asxp(self[cidx[0]])
                else:
                    assert method == "2site"
                    cms = tensordot(self[cidx[0]], self[cidx[1]], axes=1)
                hop = hop_expr(ltensor, rtensor, cmo, cms.shape)
                cout = hop(cms)
                # clean up the elements which do not meet the qn requirements
                cout[qnmat!=mps.qntot] = 0
                mps._update_mps(cout, cidx, qnbigl, qnbigr, mmax, percent)
                if mps.compress_config.ofs is not None:
                    # need to swap the original MPS. Tedious to implement and probably not useful.
                    raise NotImplementedError("OFS for variational compress not implemented")

            mps._switch_direction()

            # check convergence
            if isweep > 0 and percent == 0:
                error = mps.distance(mps_old) / np.sqrt(mps.dot(mps.conj()).real)
                logger.info(f"Variation compress relative error: {error}")
                if error < mps.compress_config.vrtol:
                    logger.info("Variational compress is converged!")
                    break

            mps_old = mps.copy()
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
        cstruct : ndarray, List[ndarray]
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
        averaged_ms :
            if ``cstruct`` is a list, ``averaged_ms`` is a list of rotated ms of
                each element in ``cstruct`` as a single site calculation. It is
                used for better initial guess in SA-DMRG algorithm. Otherwise,
                ``None`` is returned.
                ``self`` is overwritten inplace.

        """

        system = "L" if self.to_right else "R"

        # step 1: get the selected U, S, V
        if type(cstruct) is not list:
            if self.compress_config.ofs is None:
                # SVD method
                # full_matrices = True here to enable increase the bond dimension
                Uset, SUset, qnlnew, Vset, SVset, qnrnew = svd_qn.svd_qn(
                    asnumpy(cstruct), qnbigl, qnbigr, self.qntot, system=system
                )
            else:
                if isinstance(self.model, HolsteinModel):
                    # the HolsteinModel class methods are incompatible with OFS
                    raise NotImplementedError("Can't perform OFS on Holstein model")

                qnbigl1, qnbigr1 = qnbigl, qnbigr
                Uset1, SUset1, qnlnew1, Vset1, SVset1, qnrnew1 = svd_qn.svd_qn(
                    asnumpy(cstruct), qnbigl1, qnbigr1, self.qntot, system=system
                )
                qnbigl2, qnbigr2, _ = self._get_big_qn(cidx, swap=True)
                if cstruct.ndim == 4:
                    cstruct2 = asnumpy(cstruct).transpose(0, 2, 1, 3)
                else:
                    assert cstruct.ndim == 6
                    cstruct2 = asnumpy(cstruct).transpose(0, 3, 4, 1, 2, 5)
                if self.compress_config.ofs_swap_jw:
                    assert cstruct2.ndim == 4
                    cstruct2 = cstruct2.copy()
                    cstruct2[:, 1, 1, :] = -cstruct2[:, 1, 1, :]
                Uset2, SUset2, qnlnew2, Vset2, SVset2, qnrnew2 = svd_qn.svd_qn(
                    cstruct2, qnbigl2, qnbigr2, self.qntot, system=system
                )
                entropy1 = calc_vn_entropy(SUset1**2)
                entropy2 = calc_vn_entropy(SUset2**2)
                loss1 = (np.sort(SUset1)[::-1][Mmax:] ** 2).sum()
                loss2 = (np.sort(SUset2)[::-1][Mmax:] ** 2).sum()
                ofs = self.compress_config.ofs
                if ofs is OFS.ofs_d:
                    should_retain = loss1 <= loss2
                elif ofs is OFS.ofs_ds:
                    if loss1 < 1e-10 and loss2 < 1e-10:
                        # at the end of the chain
                        should_retain = entropy1 <= entropy2
                    else:
                        should_retain = loss1 <= loss2
                elif ofs is OFS.ofs_s:
                    should_retain = entropy1 <= entropy2
                else:
                    assert ofs is  OFS.ofs_debug
                    should_retain = True
                logger.debug(f"OFS: site index {cidx}, should swap: {not should_retain}, "
                             f"S: {entropy1}, {entropy2}, loss: {loss1}, {loss2}")
                if should_retain:
                    Uset, SUset, qnlnew, Vset, SVset, qnrnew = \
                        Uset1, SUset1, qnlnew1, Vset1, SVset1, qnrnew1
                else:
                    Uset, SUset, qnlnew, Vset, SVset, qnrnew = \
                        Uset2, SUset2, qnlnew2, Vset2, SVset2, qnrnew2
                    qnbigl, qnbigr, cstruct = qnbigl2, qnbigr2, cstruct2
                    new_basis = self.model.basis.copy()
                    new_basis[cidx[0]:cidx[1] + 1] = reversed(self.model.basis[cidx[0]:cidx[1] + 1])
                    # previously cached MPOs are destroyed.
                    # Not sure what is the best way: swap all cached MPOs or simply reconstruct them
                    # Need some additional testing at production level calculation
                    self.model: Model = Model(new_basis, self.model.ham_terms, self.model.dipole, self.model.output_ordering)
                logger.debug(f"DOF ordering: {[b.dof for b in self.model.basis]}")

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
            Uset, Sset, qnnew = svd_qn.eigh_qn(
                asnumpy(ddm), qnbigl, qnbigr, self.qntot, system=system
            )
            ms, msdim, msqn, compms = select_basis(
                Uset, Sset, qnnew, None, Mmax, percent=percent
            )
            rotated_c = []
            averaged_ms = []
            if self.to_right:
                ms = ms.reshape(list(qnbigl.shape) + [msdim])
                for c in cstruct:
                    compms = tensordot(
                            ms,
                            c,
                            axes=(range(qnbigl.ndim), range(qnbigl.ndim)),
                            )
                    rotated_c.append(compms)
                compms = rotated_c[0]
            else:
                ms = ms.reshape(list(qnbigr.shape) + [msdim])
                for c in cstruct:
                    compms = tensordot(
                            c,
                            ms,
                            axes=(range(qnbigl.ndim, cstruct[0].ndim), range(qnbigr.ndim)),
                            )
                    rotated_c.append(compms)
                compms = rotated_c[0]
                ms = xp.moveaxis(ms, -1, 0)

        # step 2, put updated U, S, V back to self
        if len(cidx) == 1:
            # 1site method
            self[cidx[0]] = ms
            if self.to_right:
                if cidx[0] != self.site_num - 1:
                    if type(cstruct) is list:
                        for c in rotated_c:
                            averaged_ms.append(tensordot(c, self[cidx[0] + 1],
                                axes=1))
                    self[cidx[0] + 1] = tensordot(compms, self[cidx[0] + 1], axes=1)
                    self.qn[cidx[0] + 1] = msqn
                    self.qnidx = cidx[0] + 1
                else:
                    if type(cstruct) is list:
                        for c in rotated_c:
                            averaged_ms.append(tensordot(self[cidx[0]], c, axes=1))
                    self[cidx[0]] = tensordot(self[cidx[0]], compms, axes=1)
                    self.qnidx = self.site_num - 1
            else:
                if cidx[0] != 0:
                    if type(cstruct) is list:
                        for c in rotated_c:
                            averaged_ms.append(tensordot(self[cidx[0] - 1], c, axes=1))
                    self[cidx[0] - 1] = tensordot(self[cidx[0] - 1], compms, axes=1)
                    self.qn[cidx[0]] = msqn
                    self.qnidx = cidx[0] - 1
                else:
                    if type(cstruct) is list:
                        for c in rotated_c:
                            averaged_ms.append(tensordot(c, self[cidx[0]], axes=1))
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
            if type(cstruct) is list:
                averaged_ms = rotated_c
            self.qn[cidx[1]] = msqn
        if type(cstruct) is list:
            return averaged_ms
        else:
            return None


    def _push_cano(self, idx):
        # move the canonical center to the next site
        # idx is the current canonical center
        mt: Matrix = self[idx]
        assert mt.any()
        qnbigl, qnbigr, _ = self._get_big_qn([idx])
        system = "L" if self.to_right else "R"
        u, qnlset, v, qnrset = svd_qn.svd_qn(
            mt.array,
            qnbigl,
            qnbigr,
            self.qntot,
            QR=True,
            system=system,
            full_matrices=False,
        )
        self._update_ms(
            idx, u, v.T, sigma=None, qnlset=qnlset, qnrset=qnrset
        )

    def canonicalise(self, stop_idx: int=None):
        # stop_idx: mix canonical site at `stop_idx`
        if self.to_right:
            assert self.qnidx == 0
        else:
            assert self.qnidx == self.site_num-1

        for idx in self.iter_idx_list(full=False, stop_idx=stop_idx):
           self._push_cano(idx)
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
        # use getitem/setitem to handle strings
        for i in range(self.site_num):
            new[i] = self[i].copy()
        return new

    # only (shallow) copy metadata because usually after been copied the real data is overwritten
    def metacopy(self) -> "MatrixProduct":
        new = self.__class__.__new__(self.__class__)
        new._mp = [None] * len(self)
        new.dtype = self.dtype
        # With OFS, `model` is a mutable object
        new.model = self.model.copy()
        # need to deep copy compress_config because threshold might change dynamically
        new.compress_config = self.compress_config.copy()
        new.qn = [qn.copy() for qn in self.qn]
        new.qnidx = self.qnidx
        new.qntot = self.qntot
        new.to_right = self.to_right
        new.compress_add = self.compress_add
        return new

    def _array2mt(self, array, idx, allow_dump=True):
        # convert dtype
        if isinstance(array, Matrix):
            mt = array.astype(self.dtype)
        else:
            mt = Matrix(array, dtype=self.dtype)
        if mt.pdim[0] != self.pbond_list[idx]:
            raise ValueError("Matrix physical bond dimension does not match system information")
        # setup the matrix
        mt.sigmaqn = self._get_sigmaqn(idx)

        # array too large. Should be stored in disk
        # use ``while`` to handle the multiple-exit logic
        while allow_dump and self.compress_config.dump_matrix_size < mt.array.nbytes:
            dir_with_id = os.path.join(self.compress_config.dump_matrix_dir, str(id(self)))
            if not os.path.exists(dir_with_id):
                try:
                    os.mkdir(dir_with_id)
                except:
                    logger.exception("Creating dump dir failed. Working with the matrix in memory.")
                    break
            dump_name = os.path.join(dir_with_id, f"{idx}.npy")
            try:
                array = mt.array
                if not array.flags.c_contiguous and not array.flags.f_contiguous:
                    # for faster dump (3x). Costs more memory.
                    array = np.ascontiguousarray(array)
                np.save(dump_name, array)
            except:
                logger.exception("Save matrix to disk failed. Working with the matrix in memory.")
                break
            return dump_name

        return mt

    def build_empty_mp(self, num):
        self._mp = [[None]] * num

    def dump(self, fname, other_attrs=None):

        if other_attrs is None:
            other_attrs = []
        elif isinstance(other_attrs, str):
            other_attrs = [other_attrs]
        assert isinstance(other_attrs, list)

        data_dict = dict()
        # version of the protocol
        data_dict["version"] = "0.3"
        data_dict["nsites"] = self.site_num
        for idx, mt in enumerate(self):
            data_dict[f"mt_{idx}"] = mt.array
        for attr in ["qn", "qnidx", "qntot", "to_right"] + other_attrs:
            data_dict[attr] = getattr(self, attr)
            # qn is ragged array which will raise a VisibleDeprecationWarning
            # and convert it to np.object
        try:
            np.savez(fname, **data_dict)
        except Exception:
            logger.exception(f"Dump MP failed.")

    @property
    def total_bytes(self):
        return sum(array.nbytes for array in self)

    def _get_sigmaqn(self, idx):
        raise NotImplementedError

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
        for i in range(self.site_num):
            yield self[i]

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
        mt_or_str_or_list = self._mp[item]
        if isinstance(mt_or_str_or_list, list):
            assert isinstance(item, slice)
            for elem in mt_or_str_or_list:
                if isinstance(elem, str):
                    # load all matrices to memory will make
                    # the dump mechanism pointless
                    raise IndexError("Can't slice on dump matrices.")
        if isinstance(mt_or_str_or_list, str):
            try:
                mt = Matrix(np.load(mt_or_str_or_list), dtype=self.dtype)
                mt.sigmaqn = self._get_sigmaqn(item)
            except:
                logger.exception(f"Can't load matrix from {mt_or_str_or_list}")
                raise RuntimeError("MPS internal structure corrupted.")
        else:
            if not isinstance(mt_or_str_or_list, (Matrix, type(None))):
                raise RuntimeError(f"Unknown matrix type: {type(mt_or_str_or_list)}")
            mt = mt_or_str_or_list
        return mt

    def __setitem__(self, key, array):
        old_mt = self._mp[key]
        if isinstance(old_mt, str):
            try:
                os.remove(old_mt)
            except:
                logger.exception(f"Remove {old_mt} failed")
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
        if self.is_mps:
            string = "mps"
        elif self.is_mpo:
            string = "mpo"
        elif self.is_mpdm:
            string = "mpdm"
        else:
            assert False
        template_str = "{} current size: {}, Matrix product bond dim:{}"

        return template_str.format(string, sizeof_fmt(self.total_bytes), self.bond_dims,)

    def __del__(self):
        dir_with_id = os.path.join(self.compress_config.dump_matrix_dir, str(id(self)))
        if os.path.exists(dir_with_id):
            try:
                shutil.rmtree(dir_with_id)
            except OSError:
                logger.exception(f"Removing temperary dump dir {dir_with_id} failed")
