# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import h5py
import tempfile
import scipy


# sort by similarity has problem which flips the ordering of eigenvalues when
# the initial guess is closed to excited state.  In this situation, function
# _sort_by_similarity may mark the excited state as the first eigenvalue and
# freeze the first eigenvalue.
SORT_EIG_BY_SIMILARITY = False
# Projecting out converged eigenvectors has problems when conv_tol is loose.
# In this situation, the converged eigenvectors may be updated in the
# following iterations.  Projecting out the converged eigenvectors may lead to
# large errors to the yet converged eigenvectors.
PROJECT_OUT_CONV_EIGS = False
# default max_memory 2000 MB
def davidson(
    aop,
    x0,
    precond,
    tol=1e-12,
    max_cycle=50,
    max_space=12,
    lindep=1e-14,
    max_memory=2000,
    dot=numpy.dot,
    callback=None,
    nroots=1,
    lessio=False,
    follow_state=False,
):
    e, x = davidson1(
        lambda xs: [aop(x) for x in xs],
        x0,
        precond,
        tol,
        max_cycle,
        max_space,
        lindep,
        max_memory,
        dot,
        callback,
        nroots,
        lessio,
        follow_state,
    )[1:]
    if nroots == 1:
        return e[0], x[0]
    else:
        return e, x


def davidson1(
    aop,
    x0,
    precond,
    tol=1e-12,
    max_cycle=50,
    max_space=12,
    lindep=1e-14,
    max_memory=2000,
    dot=numpy.dot,
    callback=None,
    nroots=1,
    lessio=False,
    follow_state=False,
):

    toloose = numpy.sqrt(tol)
    # print('tol %g  toloose %g', tol, toloose)

    if (not isinstance(x0, list)) and x0.ndim == 1:
        x0 = [x0]
    # max_cycle = min(max_cycle, x0[0].size)
    max_space = min(max_space + nroots * 3, max(1,x0[0].size//2))
    # max_space*2 for holding ax and xs, nroots*2 for holding axt and xt
    _incore = max_memory * 1e6 / x0[0].nbytes > max_space * 2 + nroots * 3
    lessio = lessio and not _incore
    # print('max_cycle %d  max_space %d  max_memory %d  incore %s',
    #           max_cycle, max_space, max_memory, _incore)
    heff = None
    fresh_start = True
    e = 0
    v = None
    conv = [False] * nroots
    emin = None

    for icyc in range(max_cycle):
        if fresh_start:
            if _incore:
                xs = []
                ax = []
            else:
                xs = _Xlist()
                ax = _Xlist()
            space = 0
            # Orthogonalize xt space because the basis of subspace xs must be orthogonal
            # but the eigenvectors x0 might not be strictly orthogonal
            xt = None
            xt, x0 = _qr(x0, dot), None
            max_dx_last = 1e9
            if SORT_EIG_BY_SIMILARITY:
                conv = numpy.array([False] * nroots)
        elif len(xt) > 1:
            xt = _qr(xt, dot)
            xt = xt[:40]  # 40 trial vectors at most

        axt = aop(xt)
        for k, xi in enumerate(xt):
            xs.append(xt[k])
            ax.append(axt[k])
        rnow = len(xt)
        head, space = space, space + rnow

        if heff is None:  # Lazy initilize heff to determine the dtype
            heff = numpy.empty(
                (max_space + nroots, max_space + nroots), dtype=ax[0].dtype
            )
        else:
            heff = numpy.asarray(heff, dtype=ax[0].dtype)

        elast = e
        vlast = v
        conv_last = conv
        for i in range(space):
            if head <= i < head + rnow:
                for k in range(i - head + 1):
                    heff[head + k, i] = dot(xt[k].conj(), axt[i - head])
                    heff[i, head + k] = heff[head + k, i].conj()
            else:
                for k in range(rnow):
                    heff[head + k, i] = dot(xt[k].conj(), ax[i])
                    heff[i, head + k] = heff[head + k, i].conj()

        w, v = scipy.linalg.eigh(heff[:space, :space])
        if SORT_EIG_BY_SIMILARITY:
            e, v = _sort_by_similarity(w, v, nroots, conv, vlast, emin)
            if elast.size != e.size:
                de = e
            else:
                de = e - elast
        else:
            e = w[:nroots]
            v = v[:, :nroots]

        x0 = _gen_x0(v, xs)
        if lessio:
            ax0 = aop(x0)
        else:
            ax0 = _gen_x0(v, ax)

        if SORT_EIG_BY_SIMILARITY:
            dx_norm = [0] * nroots
            xt = [None] * nroots
            for k, ek in enumerate(e):
                if not conv[k]:
                    xt[k] = ax0[k] - ek * x0[k]
                    dx_norm[k] = numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
                    if abs(de[k]) < tol and dx_norm[k] < toloose:
                        # print('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                        #          k, dx_norm[k], ek, de[k])
                        conv[k] = True
        else:
            elast, conv_last = _sort_elast(elast, conv_last, vlast, v, fresh_start)
            de = e - elast
            dx_norm = []
            xt = []
            conv = [False] * nroots
            for k, ek in enumerate(e):
                xt.append(ax0[k] - ek * x0[k])
                dx_norm.append(numpy.sqrt(dot(xt[k].conj(), xt[k]).real))
                conv[k] = abs(de[k]) < tol and dx_norm[k] < toloose
                # if conv[k] and not conv_last[k]:
                #    print('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                #              k, dx_norm[k], ek, de[k])
        ax0 = None
        max_dx_norm = max(dx_norm)
        ide = numpy.argmax(abs(de))
        if all(conv):
            # print('converge %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
            #          icyc, space, max_dx_norm, e, de[ide])
            break
        elif (
            follow_state
            and max_dx_norm > 1
            and max_dx_norm / max_dx_last > 3
            and space > nroots * 1
        ):
            # print('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
            #          icyc, space, max_dx_norm, e, de[ide], norm_min)
            # print('Large |r| detected, restore to previous x0')
            x0 = _gen_x0(vlast, xs)
            fresh_start = True
            continue

        if SORT_EIG_BY_SIMILARITY:
            if any(conv) and e.dtype == numpy.double:
                emin = min(e)

        # remove subspace linear dependency
        if any(((not conv[k]) and n ** 2 > lindep) for k, n in enumerate(dx_norm)):
            for k, ek in enumerate(e):
                if (not conv[k]) and dx_norm[k] ** 2 > lindep:
                    xt[k] = precond(xt[k], e[0], x0[k])
                    xt[k] *= 1 / numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
                else:
                    xt[k] = None
        else:
            for k, ek in enumerate(e):
                if dx_norm[k] ** 2 > lindep:
                    xt[k] = precond(xt[k], e[0], x0[k])
                    xt[k] *= 1 / numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
                else:
                    xt[k] = None
        xt = [xi for xi in xt if xi is not None]

        for i in range(space):
            xsi = xs[i]
            for xi in xt:
                xi -= xsi * dot(xsi.conj(), xi)
        norm_min = 1
        for i, xi in enumerate(xt):
            norm = numpy.sqrt(dot(xi.conj(), xi).real)
            if norm ** 2 > lindep:
                xt[i] *= 1 / norm
                norm_min = min(norm_min, norm)
            else:
                xt[i] = None
        xt = [xi for xi in xt if xi is not None]
        xi = None
        # print('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
        #          icyc, space, max_dx_norm, e, de[ide], norm_min)
        if len(xt) == 0:
            # print('Linear dependency in trial subspace. |r| for each state %s', dx_norm)
            conv = [conv[k] or (norm < toloose) for k, norm in enumerate(dx_norm)]
            break

        max_dx_last = max_dx_norm
        fresh_start = space + nroots > max_space

        if callable(callback):
            callback(locals())

    return conv, e, x0


def _qr(xs, dot):
    norm = numpy.sqrt(dot(xs[0].conj(), xs[0]).real)
    qs = [xs[0] / norm]
    for i in range(1, len(xs)):
        xi = xs[i].copy()
        for j in range(len(qs)):
            xi -= qs[j] * dot(qs[j].conj(), xi)
        norm = numpy.sqrt(dot(xi.conj(), xi).real)
        if norm > 1e-7:
            qs.append(xi / norm)
    return qs


def _gen_x0(v, xs):
    space, nroots = v.shape
    x0 = []
    for k in range(nroots):
        x0.append(xs[space - 1] * v[space - 1, k])
    for i in reversed(range(space - 1)):
        xsi = xs[i]
        for k in range(nroots):
            x0[k] += v[i, k] * xsi
    return x0


def _sort_by_similarity(w, v, nroots, conv, vlast, emin=None, heff=None):
    if not any(conv) or vlast is None:
        return w[:nroots], v[:, :nroots]

    head, nroots = vlast.shape
    conv = numpy.asarray(conv[:nroots])
    ovlp = vlast[:, conv].T.conj().dot(v[:head])
    ovlp = numpy.einsum("ij,ij->j", ovlp, ovlp)
    nconv = numpy.count_nonzero(conv)
    nleft = nroots - nconv
    idx = ovlp.argsort()
    sorted_idx = numpy.zeros(nroots, dtype=int)
    sorted_idx[conv] = numpy.sort(idx[-nconv:])
    sorted_idx[~conv] = numpy.sort(idx[:-nconv])[:nleft]

    e = w[sorted_idx]
    c = v[:, sorted_idx]
    return e, c


def _sort_elast(elast, conv_last, vlast, v, fresh_start):
    if fresh_start:
        return elast, conv_last
    head, nroots = vlast.shape
    ovlp = abs(numpy.dot(v[:head].conj().T, vlast))
    idx = numpy.argmax(ovlp, axis=1)
    return [elast[i] for i in idx], [conv_last[i] for i in idx]


class H5TmpFile(h5py.File):
    def __init__(self, filename=None, *args, **kwargs):
        if filename is None:
            tmpfile = tempfile.NamedTemporaryFile(dir=".")
            filename = tmpfile.name
        h5py.File.__init__(self, filename, *args, **kwargs)

    def __del__(self):
        self.close()


class _Xlist(list):
    def __init__(self):
        self.scr_h5 = H5TmpFile()
        self.index = []

    def __getitem__(self, n):
        key = self.index[n]
        return self.scr_h5[key].value

    def append(self, x):
        key = str(len(self.index) + 1)
        if key in self.index:
            for i in range(len(self.index) + 1):
                if str(i) not in self.index:
                    key = str(i)
                    break
        self.index.append(key)
        self.scr_h5[key] = x
        self.scr_h5.flush()

    def __setitem__(self, n, x):
        key = self.index[n]
        self.scr_h5[key][:] = x
        self.scr_h5.flush()

    def __len__(self):
        return len(self.index)

    def pop(self, index):
        key = self.index.pop(index)
        del (self.scr_h5[key])
