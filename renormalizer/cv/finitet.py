# -*- coding: utf-8 -*-
# Author: Tong Jiang <tongjiang1000@gmail.com>
# finite temperature absorption/emission spectrum based on Correction vector

import copy
import os
import logging
import scipy
from itertools import product

from renormalizer.mps.matrix import (
    multi_tensor_contract,
    tensordot,
    moveaxis
)
from renormalizer.cv.spectra_cv import SpectraCv
from renormalizer.mps.backend import np, xp
from renormalizer.mps.matrix import asxp, asnumpy
from renormalizer.mps import (
    Mpo, svd_qn, MpDm, ThermalProp, load_thermal_state)
from renormalizer.mps.lib import update_cv
from renormalizer.utils import (
    CompressConfig, EvolveConfig,
    CompressCriteria
)

logger = logging.getLogger(__name__)


class SpectraFtCV(SpectraCv):
    r"""
    Use DDMRG to calculate the finite temperature spectrum from frequency domain

    Args:
        model (:class:`~renormalizer.model.Model`): system information.
        spectratype (string): "abs" or "emi".
        m_max (int): maximal bond dimension of correction vector.
        eta (float): Lorentzian broadening width (a.u.).
        temperature (:class:`~renormalizer.utils.Quantity`): simulation temperature.
        h_mpo (:class:`~renormalizer.mps.Mpo`): system Hamiltonian.
        method (str): "1site" or "2site".
        procedure_cv (list): percent used for each sweep.
        rtol (float): the relative tolerance of the spectrum strength, default: 1e-5.
        b_mps (:class:`~renormalizer.mps.Mps`): the b vector -eta * dipole * \psi_0, default: None.
            (Holstein model could construct b_mps implicitly).
        cv_mps (:class:`~renormalizer.mps.Mps`): initial guess of cv_mps, default: None.
        icompress_config (:class:`~renormalizer.utils.CompressConfig`): config when compressing MPS/MPO during the imaginary time evolution..
        ievolve_config (:class:`~renormalizer.utils.EvolveConfig`): evolution config for imaginary time evolution.
        insteps (int): evolve steps for imaginary time evolution. have to be provided when calculating emission.
        dump_dir (str): the directory for logging and numerical result output.
            Also the directory from which to load previous thermal propagated initial state (if exists).
        job_name (str): the name of the calculation job which determines the file name of the logging and numerical result output.

    Example::
        see test/test_abs.py for example

    """
    def __init__(
        self,
        model,
        spectratype,
        m_max,
        eta,
        temperature,
        h_mpo=None,
        method='1site',
        procedure_cv=None,
        rtol=1e-5,
        b_mps=None,
        cv_mps=None,
        icompress_config=None,
        ievolve_config=None,
        insteps=None,
        dump_dir: str=None,
        job_name=None,
    ):

        self.temperature = temperature
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
        self.job_name = job_name
        self.dump_dir = dump_dir

        super().__init__(
            model, spectratype, m_max, eta, h_mpo=h_mpo,
            method=method, procedure_cv=procedure_cv,
            rtol=rtol, b_mps=b_mps, cv_mps=cv_mps,
        )

        self.cv_mpo = self.cv_mps
        self.b_mpo = self.b_mps

        self.a_oper = None

    def init_cv_mpo(self):
        cv_mpo = Mpo.finiteT_cv(self.model, 1, self.m_max,
                                self.spectratype, percent=1.0)
        return cv_mpo

    init_cv_mps = init_cv_mpo

    def init_b_mpo(self):
        # get the right hand site vector b, Ax=b
        # b = -eta * dipole * \psi_0

        # only support Holstien model 0/1 exciton manifold
        beta = self.temperature.to_beta()
        if self.spectratype == "abs":
            dipole_mpo = Mpo.onsite(self.model, r"a^\dagger", dipole=True)
            i_mpo = MpDm.max_entangled_gs(self.model)
            tp = ThermalProp(i_mpo, exact=True, space='GS')
            tp.evolve(None, 1, beta / 2j)
            ket_mpo = tp.latest_mps
        elif self.spectratype == "emi":
            dipole_mpo = Mpo.onsite(self.model, "a", dipole=True)
            if self._defined_output_path:
                ket_mpo = \
                    load_thermal_state(self.model, self._thermal_dump_path)
            else:
                ket_mpo = None
            if ket_mpo is None:
                impo = MpDm.max_entangled_ex(self.model)
                impo.compress_config = self.compress_config
                if self.job_name is None:
                    job_name = None
                else:
                    job_name = self.job_name + "_thermal_prop"
                tp = ThermalProp(
                    impo, evolve_config=self.evolve_config,
                    dump_dir=self.dump_dir, job_name=job_name)
                tp.evolve(None, self.insteps, beta / 2j)
                ket_mpo = tp.latest_mps
                if self._defined_output_path:
                    ket_mpo.dump(self._thermal_dump_path)
        else:
            assert False
        ket_mpo = dipole_mpo.apply(ket_mpo.scale(-self.eta))

        return ket_mpo, None

    init_b_mps = init_b_mpo

    @property
    def _thermal_dump_path(self):
        assert self._defined_output_path
        return os.path.join(self.dump_dir, self.job_name + "_impo.npz")

    @property
    def _defined_output_path(self):
        return self.dump_dir is not None and self.job_name is not None

    def oper_prepare(self, omega):
        identity = Mpo.identity(self.model).scale(omega)
        self.a_oper = identity.add(self.h_mpo.scale(-1, inplace=False))

    def optimize_cv(self, lr_group, isite, percent=0):
        if self.spectratype == "abs":
            # quantum number restriction, |1><0|
            up_exciton, down_exciton = 1, 0
        elif self.spectratype == "emi":
            # quantum number restriction, |0><1|
            up_exciton, down_exciton = 0, 1
        nexciton = 1
        first_LR, second_LR, third_LR, forth_LR = lr_group

        if self.method == "1site":
            add_list = [isite - 1]
            first_L = asxp(first_LR[isite - 1])
            first_R = asxp(first_LR[isite])
            second_L = asxp(second_LR[isite - 1])
            second_R = asxp(second_LR[isite])
            third_L = asxp(third_LR[isite - 1])
            third_R = asxp(third_LR[isite])
            forth_L = asxp(forth_LR[isite - 1])
            forth_R = asxp(forth_LR[isite])
        else:
            add_list = [isite - 2, isite - 1]
            first_L = asxp(first_LR[isite - 2])
            first_R = asxp(first_LR[isite])
            second_L = asxp(second_LR[isite - 2])
            second_R = asxp(second_LR[isite])
            third_L = asxp(third_LR[isite - 2])
            third_R = asxp(third_LR[isite])
            forth_L = asxp(forth_LR[isite - 2])
            forth_R = asxp(forth_LR[isite])

        xqnmat, xqnbigl, xqnbigr, xshape = \
            self.construct_X_qnmat(add_list)
        dag_qnmat, dag_qnbigl, dag_qnbigr = self.swap(xqnmat, xqnbigl, xqnbigr)
        nonzeros = int(np.sum(
            self.condition(
                dag_qnmat, [down_exciton, up_exciton])
        ))

        if self.method == "1site":
            guess = moveaxis(self.cv_mpo[isite - 1], (1, 2), (2, 1))
        else:
            guess = tensordot(moveaxis(self.cv_mpo[isite - 2], (1, 2), (2, 1)),
                              moveaxis(self.cv_mpo[isite - 1]), axes=(-1, 0))
        guess = guess[
            self.condition(
                dag_qnmat, [down_exciton, up_exciton])]

        if self.method == "1site":
            # define dot path
            path_1 = [([0, 1], "abcd, aefg -> bcdefg"),
                      ([3, 0], "bcdefg, bfhi -> cdeghi"),
                      ([2, 0], "cdeghi, chjk -> degijk"),
                      ([1, 0], "degijk, gikl -> dejl")]
            path_2 = [([0, 1], "abcd, aefg -> bcdefg"),
                      ([3, 0], "bcdefg, bfhi -> cdeghi"),
                      ([2, 0], "cdeghi, djek -> cghijk"),
                      ([1, 0], "cghijk, gilk -> chjl")]
            path_3 = [([0, 1], "ab, acde -> bcde"),
                      ([1, 0], "bcde, ef -> bcdf")]

            vecb = multi_tensor_contract(
                path_3, forth_L,
                moveaxis(self.b_mpo[isite - 1], (1, 2), (2, 1)),
                forth_R)[self.condition(dag_qnmat, [down_exciton, up_exciton])]

        a_oper_isite = asxp(self.a_oper[isite - 1])
        h_mpo_isite = asxp(self.h_mpo[isite - 1])
        # construct preconditioner
        Idt = xp.identity(h_mpo_isite.shape[1])
        M1_1 = xp.einsum('abca->abc', first_L)
        path_m1 = [([0, 1], "abc, bdef->acdef"),
                   ([1, 0], "acdef, cegh->adfgh")]
        M1_2 = multi_tensor_contract(path_m1, M1_1, a_oper_isite, a_oper_isite)
        M1_2 = xp.einsum("abcbd->abcd", M1_2)
        M1_3 = xp.einsum('ecde->ecd', first_R)
        M1_4 = xp.einsum('ff->f', Idt)
        path_m1 = [([0, 1], "abcd,ecd->abe"),
                   ([1, 0], "abe,f->abef")]
        pre_M1 = multi_tensor_contract(
            path_m1, M1_2, M1_3, M1_4)
        pre_M1 = xp.moveaxis(pre_M1, [-2, -1], [-1, -2])[
            self.condition(dag_qnmat, [down_exciton, up_exciton])]

        M2_1 = xp.einsum('aeag->aeg', second_L)
        M2_2 = xp.einsum('eccf->ecf', a_oper_isite)
        M2_3 = xp.einsum('gbbh->gbh', h_mpo_isite)
        M2_4 = xp.einsum('dfdh->dfh', second_R)
        path_m2 = [([0, 1], "aeg,gbh->aebh"),
                   ([2, 0], "aebh,ecf->abchf"),
                   ([1, 0], "abhcf,dfh->abcd")]
        pre_M2 = multi_tensor_contract(
            path_m2, M2_1, M2_3, M2_2, M2_4)
        pre_M2 = pre_M2[
            self.condition(dag_qnmat, [down_exciton, up_exciton])]

        M4_1 = xp.einsum('faah->fah', third_L)
        M4_4 = xp.einsum('gddi->gdi', third_R)
        M4_5 = xp.einsum('cc->c', Idt)
        M4_path = [([0, 1], "fah,febg->ahebg"),
                   ([2, 0], "ahebg,hjei->abgji"),
                   ([1, 0], "abgji,gdi->abjd")]
        pre_M4 = multi_tensor_contract(
            M4_path, M4_1, h_mpo_isite,
            h_mpo_isite, M4_4)
        pre_M4 = xp.einsum('abbd->abd', pre_M4)
        pre_M4 = xp.tensordot(pre_M4, M4_5, axes=0)
        pre_M4 = xp.moveaxis(pre_M4, [2, 3], [3, 2])[
            self.condition(dag_qnmat, [down_exciton, up_exciton])]

        M_x = lambda x: asnumpy(asxp(x) / (pre_M1 + 2 * pre_M2 + pre_M4 + xp.ones(nonzeros)*self.eta**2))
        pre_M = scipy.sparse.linalg.LinearOperator((nonzeros, nonzeros), M_x)

        count = 0

        def hop(x):
            nonlocal count
            count += 1
            dag_struct = asxp(self.dag2mat(
                xshape, x, dag_qnmat))
            if self.method == "1site":

                M1 = multi_tensor_contract(
                    path_1, first_L, dag_struct,
                    a_oper_isite, a_oper_isite, first_R)
                M2 = multi_tensor_contract(
                    path_2, second_L, dag_struct,
                    a_oper_isite,
                    h_mpo_isite, second_R)
                M2 = xp.moveaxis(M2, (1, 2), (2, 1))
                M3 = multi_tensor_contract(
                    path_2, third_L, h_mpo_isite, dag_struct,
                    h_mpo_isite, third_R)
                M3 = xp.moveaxis(M3, (1, 2), (2, 1))
                cout = M1 + 2 * M2 + M3 + dag_struct * self.eta**2
            cout = cout[
                self.condition(dag_qnmat, [down_exciton, up_exciton])
            ]
            return asnumpy(cout)

        # Matrix A
        mat_a = scipy.sparse.linalg.LinearOperator((nonzeros, nonzeros), matvec=hop)

        x, info = scipy.sparse.linalg.cg(
            mat_a, asnumpy(vecb), tol=1.e-5, x0=asnumpy(guess), maxiter=500,
            M=pre_M, atol=0)
        # logger.info(f"linear eq dim: {nonzeros}")
        # logger.info(f'times for hop:{count}')
        self.hop_time.append(count)
        if info != 0:
            logger.warning(
                f"cg not converged, vecb.norm:{xp.linalg.norm(vecb)}")
        l_value = xp.dot(asxp(hop(x)), asxp(x)) - 2 * xp.dot(vecb, asxp(x))

        x = self.dag2mat(xshape, x, dag_qnmat)
        if self.method == "1site":
            x = np.moveaxis(x, [1, 2], [2, 1])
        x, xdim, xqn, compx = self.x_svd(
            x, xqnbigl, xqnbigr, nexciton,
            percent=percent)

        if self.method == "1site":
            self.cv_mpo[isite - 1] = x
            if not self.cv_mpo.to_right:
                if isite != 1:
                    self.cv_mpo[isite - 2] = \
                        tensordot(self.cv_mpo[isite - 2], compx, axes=(-1, 0))
                    self.cv_mpo.qn[isite - 1] = xqn
                    self.cv_mpo.qnidx = isite-2
                else:
                    self.cv_mpo[isite - 1] = \
                        tensordot(compx, self.cv_mpo[isite - 1], axes=(-1, 0))
                    self.cv_mpo.qnidx = 0
            else:
                if isite != len(self.cv_mpo):
                    self.cv_mpo[isite] = \
                        tensordot(compx, self.cv_mpo[isite], axes=(-1, 0))
                    self.cv_mpo.qn[isite] = xqn
                    self.cv_mpo.qnidx = isite
                else:
                    self.cv_mpo[isite - 1] = \
                        tensordot(self.cv_mpo[isite - 1], compx, axes=(-1, 0))
                    self.cv_mpo.qnidx = self.cv_mpo.site_num-1

        else:
            if not self.cv_mpo.to_right:
                self.cv_mpo[isite - 2] = compx
                self.cv_mpo[isite - 1] = x
                self.cv_mpo.qnidx = isite-2
            else:
                self.cv_mpo[isite - 2] = x
                self.cv_mpo[isite - 1] = compx
                self.cv_mpo.qnidx = isite-1
            self.cv_mpo.qn[isite - 1] = xqn

        return float(l_value)

    def construct_X_qnmat(self, addlist):

        pbond = self.model.pbond_list
        xqnl = np.array(self.cv_mpo.qn[addlist[0]])
        xqnr = np.array(self.cv_mpo.qn[addlist[-1] + 1])
        xqnmat = xqnl.copy()
        xqnsigmalist = []
        for idx in addlist:
            sigmaqn = self.model.basis[idx].sigmaqn
            xqnsigma = np.array(list(product(sigmaqn, repeat=2)))
            xqnsigma = xqnsigma.reshape(pbond[idx], pbond[idx], 2)

            xqnmat = self.qnmat_add(xqnmat, xqnsigma)
            xqnsigmalist.append(xqnsigma)

        xqnmat = self.qnmat_add(xqnmat, xqnr)
        matshape = list(xqnmat.shape)
        if self.method == "1site":
            if xqnmat.ndim == 4:
                if not self.cv_mpo.to_right:
                    xqnmat = np.moveaxis(xqnmat.reshape(matshape+[1]), -1, -2)
                else:
                    xqnmat = xqnmat.reshape([1] + matshape)
            if not self.cv_mpo.to_right:
                xqnbigl = xqnl.copy()
                xqnbigr = self.qnmat_add(xqnsigmalist[0], xqnr)
                if xqnbigr.ndim == 3:
                    rshape = list(xqnbigr.shape)
                    xqnbigr = np.moveaxis(xqnbigr.reshape(rshape+[1]), -1, -2)
            else:
                xqnbigl = self.qnmat_add(xqnl, xqnsigmalist[0])
                xqnbigr = xqnr.copy()
                if xqnbigl.ndim == 3:
                    lshape = list(xqnbigl.shape)
                    xqnbigl = xqnbigl.reshape([1] + lshape)
        else:
            if xqnmat.ndim == 6:
                if addlist[0] != 0:
                    xqnmat = np.moveaxis(xqnmat.resahpe(matshape+[1]), -1, -2)
                else:
                    xqnmat = xqnmat.reshape([1] + matshape)
            xqnbigl = self.qnmat_add(xqnl, xqnsigmalist[0])
            if xqnbigl.ndim == 3:
                lshape = list(xqnbigl.shape)
                xqnbigl = xqnbigl.reshape([1] + lshape)
            xqnbigr = self.qnmat_add(xqnsigmalist[-1], xqnr)
            if xqnbigr.ndim == 3:
                rshape = list(xqnbigr.shape)
                xqnbigr = np.moveaxis(xqnbigr.reshape(rshape + [1]), -1, -2)
        xshape = list(xqnmat.shape)
        del xshape[-1]
        if len(xshape) == 3:
            if not self.cv_mpo.to_right:
                xshape = xshape + [1]
            else:
                xshape = [1] + xshape
        return xqnmat, xqnbigl, xqnbigr, xshape

    def swap(self, mat, qnbigl, qnbigr):

        def inter_change(ori_mat):
            matshape = ori_mat.shape
            len_mat = int(np.prod(np.array(matshape[:-1])))
            ori_mat = ori_mat.reshape(len_mat, 2)
            change_mat = copy.deepcopy(ori_mat)
            change_mat[:, 0], change_mat[:, 1] = ori_mat[:, 1], ori_mat[:, 0]
            return change_mat.reshape(matshape)

        dag_qnmat = inter_change(mat)
        if self.method == "1site":
            dag_qnmat = np.moveaxis(dag_qnmat, [1, 2], [2, 1])
            dag_qnbigl = inter_change(qnbigl)
            dag_qnbigr = inter_change(qnbigr)
            if not self.cv_mpo.to_right:
                dag_qnbigr = np.moveaxis(dag_qnbigr, [0, 1], [1, 0])
            else:
                dag_qnbigl = np.moveaxis(dag_qnbigl, [1, 2], [2, 1])
        else:
            raise NotImplementedError
            # we don't recommend 2-site CV-DMRG, which is a huge cost

        return dag_qnmat, dag_qnbigl, dag_qnbigr

    def condition(self, mat, qn):
        condition = (mat == qn)
        mat_shape = list(condition.shape)
        del mat_shape[-1]
        condition = condition.all(axis=-1)
        condition = condition.reshape(mat_shape)
        return condition

    def qnmat_add(self, mat_l, mat_r):
        lshape, rshape = mat_l.shape, mat_r.shape
        lena = int(np.prod(np.array(lshape)) / 2)
        lenb = int(np.prod(np.array(rshape)) / 2)
        matl = mat_l.reshape(lena, 2)
        matr = mat_r.reshape(lenb, 2)
        lr1 = np.add.outer(matl[:, 0], matr[:, 0]).flatten()
        lr2 = np.add.outer(matl[:, 1], matr[:, 1]).flatten()
        lr = np.zeros((len(lr1), 2))
        lr[:, 0] = lr1
        lr[:, 1] = lr2
        shapel = list(mat_l.shape)
        del shapel[-1]
        shaper = list(mat_r.shape)
        del shaper[-1]
        lr = lr.reshape(shapel + shaper + [2])
        return lr

    def dag2mat(self, xshape, x, dag_qnmat):
        if self.spectratype == "abs":
            up_exciton, down_exciton = 1, 0
        else:
            up_exciton, down_exciton = 0, 1
        xdag = np.zeros(xshape, dtype=x.dtype)
        mask = self.condition(dag_qnmat, [down_exciton, up_exciton])
        np.place(xdag, mask, x)
        shape = list(xdag.shape)
        if xdag.ndim == 3:
            if not self.cv_mpo.to_right:
                xdag = xdag.reshape(shape + [1])
            else:
                xdag = xdag.reshape([1] + shape)
        return xdag

    def x_svd(self, xstruct, xqnbigl, xqnbigr, nexciton, percent=0):
        Gamma = xstruct.reshape(
            np.prod(xqnbigl.shape) // 2, np.prod(xqnbigr.shape) // 2)

        localXqnl = xqnbigl.ravel()
        localXqnr = xqnbigr.ravel()
        list_locall = []
        list_localr = []
        for i in range(0, len(localXqnl), 2):
            list_locall.append([localXqnl[i], localXqnl[i + 1]])
        for i in range(0, len(localXqnr), 2):
            list_localr.append([localXqnr[i], localXqnr[i + 1]])
        localXqnl = copy.deepcopy(list_locall)
        localXqnr = copy.deepcopy(list_localr)
        xuset = []
        xuset0 = []
        xvset = []
        xvset0 = []
        xsset = []
        xsuset0 = []
        xsvset0 = []
        xqnlset = []
        xqnlset0 = []
        xqnrset = []
        xqnrset0 = []
        if self.spectratype == "abs":
            combine = [[[y, 0], [nexciton - y, 0]]
                       for y in range(nexciton + 1)]
        elif self.spectratype == "emi":
            combine = [[[0, y], [0, nexciton - y]]
                       for y in range(nexciton + 1)]
        for nl, nr in combine:
            lset = np.where(self.condition(np.array(localXqnl), [nl]))[0]
            rset = np.where(self.condition(np.array(localXqnr), [nr]))[0]
            if len(lset) != 0 and len(rset) != 0:
                Gamma_block = Gamma.ravel().take(
                    (lset * Gamma.shape[1]).reshape(-1, 1) + rset)
                try:
                    U, S, Vt = \
                        scipy.linalg.svd(Gamma_block, full_matrices=True,
                                         lapack_driver='gesdd')
                except:
                    U, S, Vt = \
                        scipy.linalg.svd(Gamma_block, full_matrices=True,
                                         lapack_driver='gesvd')

                dim = S.shape[0]

                xsset.append(S)
                # U part quantum number
                xuset.append(
                    svd_qn.blockrecover(lset, U[:, :dim], Gamma.shape[0]))
                xqnlset += [nl] * dim
                xuset0.append(
                    svd_qn.blockrecover(lset, U[:, dim:], Gamma.shape[0]))
                xqnlset0 += [nl] * (U.shape[0] - dim)
                xsuset0.append(np.zeros(U.shape[0] - dim))
                # V part quantum number
                VT = Vt.T
                xvset.append(
                    svd_qn.blockrecover(rset, VT[:, :dim], Gamma.shape[1]))
                xqnrset += [nr] * dim
                xvset0.append(
                    svd_qn.blockrecover(rset, VT[:, dim:], Gamma.shape[1]))
                xqnrset0 += [nr] * (VT.shape[0] - dim)
                xsvset0.append(np.zeros(VT.shape[0] - dim))
        xuset = np.concatenate(xuset + xuset0, axis=1)
        xvset = np.concatenate(xvset + xvset0, axis=1)
        xsuset = np.concatenate(xsset + xsuset0)
        xsvset = np.concatenate(xsset + xsvset0)
        xqnlset = xqnlset + xqnlset0
        xqnrset = xqnrset + xqnrset0
        bigl_shape = list(xqnbigl.shape)
        del bigl_shape[-1]
        bigr_shape = list(xqnbigr.shape)
        del bigr_shape[-1]
        if not self.cv_mpo.to_right:
            x, xdim, xqn, compx = update_cv(
                xvset, xsvset, xqnrset, xuset, nexciton, self.m_max,
                self.spectratype, percent=percent)
            if (self.method == "1site") and (len(bigr_shape + [xdim]) == 3):
                return np.moveaxis(
                    x.reshape(bigr_shape + [1] + [xdim]), -1, 0),\
                    xdim, xqn, compx.reshape(bigl_shape + [xdim])
            else:
                return np.moveaxis(x.reshape(bigr_shape + [xdim]), -1, 0),\
                    xdim, xqn, compx.reshape(bigl_shape + [xdim])
        else:
            x, xdim, xqn, compx = update_cv(
                xuset, xsuset, xqnlset, xvset, nexciton,
                self.m_max, self.spectratype, percent=percent)
            if (self.method == "1site") and (len(bigl_shape + [xdim]) == 3):
                return x.reshape([1] + bigl_shape + [xdim]), xdim, xqn, \
                    np.moveaxis(compx.reshape(bigr_shape + [xdim]), -1, 0)
            else:
                return x.reshape(bigl_shape + [xdim]), xdim, xqn, \
                    np.moveaxis(compx.reshape(bigr_shape + [xdim]), -1, 0)

    def initialize_LR(self):

        first_LR = [np.ones((1, 1, 1, 1))]
        forth_LR = [np.ones((1, 1))]
        for isite in range(1, len(self.cv_mpo)):
            first_LR.append(None)
            forth_LR.append(None)
        first_LR.append(np.ones((1, 1, 1, 1)))
        second_LR = copy.deepcopy(first_LR)
        third_LR = copy.deepcopy(first_LR)
        forth_LR.append(np.ones((1, 1)))

        if self.cv_mpo.to_right:
            for isite in range(len(self.cv_mpo), 1, -1):
                cv_isite = self.cv_mpo[isite-1]
                dag_cv_isite = moveaxis(cv_isite, (1, 2), (2, 1))
                path1 = [([0, 1], "abcd, efga -> bcdefg"),
                         ([3, 0], "bcdefg, hgib -> cdefhi"),
                         ([2, 0], "cdefhi, jikc -> defhjk"),
                         ([1, 0], "defhjk, lkfd -> ehjl")]
                path2 = [([0, 1], "ab, cdea->bcde"),
                         ([1, 0], "bcde, fedb->cf")]
                first_LR[isite - 1] = asnumpy(multi_tensor_contract(
                    path1, first_LR[isite],
                    dag_cv_isite, self.a_oper[isite - 1],
                    self.a_oper[isite - 1], cv_isite))

                second_LR[isite - 1] = asnumpy(multi_tensor_contract(
                    path1, second_LR[isite],
                    dag_cv_isite,
                    self.a_oper[isite - 1], cv_isite,
                    self.h_mpo[isite - 1]))
                third_LR[isite - 1] = asnumpy(multi_tensor_contract(
                    path1, third_LR[isite], self.h_mpo[isite - 1],
                    dag_cv_isite,
                    cv_isite, self.h_mpo[isite - 1]))
                forth_LR[isite - 1] = asnumpy(multi_tensor_contract(
                    path2, forth_LR[isite],
                    moveaxis(self.b_mpo[isite - 1], (1, 2), (2, 1)),
                    cv_isite))

        else:
            for isite in range(1, len(self.cv_mpo)):
                cv_isite = self.cv_mpo[isite-1]
                dag_cv_isite = moveaxis(cv_isite, (1, 2), (2, 1))
                path1 = [([0, 1], "abcd, aefg -> bcdefg"),
                         ([3, 0], "bcdefg, bfhi -> cdeghi"),
                         ([2, 0], "cdeghi, chjk -> degijk"),
                         ([1, 0], "degijk, djel -> gikl")]
                path2 = [([0, 1], "ab, acde->bcde"),
                         ([1, 0], "bcde, bdcf->ef")]
                first_LR[isite] = asnumpy(multi_tensor_contract(
                    path1, first_LR[isite - 1],
                    dag_cv_isite, self.a_oper[isite - 1],
                    self.a_oper[isite - 1], cv_isite))
                second_LR[isite] = asnumpy(multi_tensor_contract(
                    path1, second_LR[isite - 1],
                    dag_cv_isite,
                    self.a_oper[isite - 1], cv_isite,
                    self.h_mpo[isite - 1]))
                third_LR[isite] = asnumpy(multi_tensor_contract(
                    path1, third_LR[isite - 1], self.h_mpo[isite - 1],
                    dag_cv_isite,
                    cv_isite, self.h_mpo[isite - 1]))
                forth_LR[isite] = asnumpy(multi_tensor_contract(
                    path2, forth_LR[isite - 1],
                    moveaxis(self.b_mpo[isite - 1], (1, 2), (2, 1)),
                    cv_isite))
        return [first_LR, second_LR, third_LR, forth_LR]

    def update_LR(self, lr_group, isite):
        first_LR, second_LR, third_LR, forth_LR = lr_group
        cv_isite = self.cv_mpo[isite-1]
        dag_cv_isite = moveaxis(cv_isite, (1, 2), (2, 1))
        if self.method == "1site":
            if not self.cv_mpo.to_right:
                path1 = [([0, 1], "abcd, efga -> bcdefg"),
                         ([3, 0], "bcdefg, hgib -> cdefhi"),
                         ([2, 0], "cdefhi, jikc -> defhjk"),
                         ([1, 0], "defhjk, lkfd -> ehjl")]
                path2 = [([0, 1], "ab, cdea->bcde"),
                         ([1, 0], "bcde, fedb->cf")]
                first_LR[isite - 1] = multi_tensor_contract(
                    path1, first_LR[isite],
                    dag_cv_isite, self.a_oper[isite - 1],
                    self.a_oper[isite - 1], cv_isite)
                second_LR[isite - 1] = multi_tensor_contract(
                    path1, second_LR[isite],
                    dag_cv_isite,
                    self.a_oper[isite - 1], cv_isite,
                    self.h_mpo[isite - 1])
                third_LR[isite - 1] = multi_tensor_contract(
                    path1, third_LR[isite], self.h_mpo[isite - 1],
                    dag_cv_isite,
                    cv_isite, self.h_mpo[isite - 1])
                forth_LR[isite - 1] = multi_tensor_contract(
                    path2, forth_LR[isite],
                    moveaxis(self.b_mpo[isite - 1], (1, 2), (2, 1)),
                    cv_isite)

            else:
                path1 = [([0, 1], "abcd, aefg -> bcdefg"),
                         ([3, 0], "bcdefg, bfhi -> cdeghi"),
                         ([2, 0], "cdeghi, chjk -> degijk"),
                         ([1, 0], "degijk, djel -> gikl")]
                path2 = [([0, 1], "ab, acde->bcde"),
                         ([1, 0], "bcde, bdcf->ef")]

                first_LR[isite] = multi_tensor_contract(
                    path1, first_LR[isite - 1],
                    dag_cv_isite,
                    self.a_oper[isite - 1], self.a_oper[isite - 1], cv_isite)

                second_LR[isite] = multi_tensor_contract(
                    path1, second_LR[isite - 1],
                    dag_cv_isite,
                    self.a_oper[isite - 1], cv_isite,
                    self.h_mpo[isite - 1])
                third_LR[isite] = multi_tensor_contract(
                    path1, third_LR[isite - 1], self.h_mpo[isite - 1],
                    dag_cv_isite,
                    cv_isite, self.h_mpo[isite - 1])
                forth_LR[isite] = multi_tensor_contract(
                    path2, forth_LR[isite - 1],
                    moveaxis(self.b_mpo[isite - 1], (1, 2), (2, 1)),
                    cv_isite)
        else:
            # 2site for finite temperature is too expensive, so I drop it
            # (at least for now)
            raise NotImplementedError

        return first_LR, second_LR, third_LR, forth_LR
