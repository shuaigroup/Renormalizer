from __future__ import division, print_function, absolute_import
from math import prod
from itertools import groupby
from warnings import warn
import numpy as np

from scipy.sparse import find, coo_matrix

from renormalizer.backend.context import internal_backend as xp


# multibackend: a host constant; no backend needed at import.
EPS = np.finfo(float).eps


def host_time_array(value):
    """Materialize only time/event metadata; numerical states stay native."""
    if xp.current.is_array(value):
        value = xp.numpy(value)
    return np.asarray(value)


def validate_first_step(first_step, t0, t_bound):
    """Assert that first_step is valid and return it."""
    if first_step <= 0:
        raise ValueError("`first_step` must be positive.")
    # Time bounds are host scalars, even when the state lives on a device.
    if first_step > abs(t_bound - t0):
        raise ValueError("`first_step` exceeds bounds.")
    return first_step


def validate_max_step(max_step):
    """Assert that max_Step is valid and return it."""
    if max_step <= 0:
        raise ValueError("`max_step` must be positive.")
    return max_step


def warn_extraneous(extraneous):
    """Display a warning for extraneous keyword arguments.

    The initializer of each solver class is expected to collect keyword
    arguments that it doesn't understand and warn about them. This function
    prints a warning for each key in the supplied dictionary.

    Parameters
    ----------
    extraneous : dict
        Extraneous keyword arguments
    """
    if extraneous:
        warn(
            "The following arguments have no effect for a chosen solver: {}.".format(
                ", ".join("`{}`".format(x) for x in extraneous)
            )
        )


def validate_tol(rtol, atol, n):
    """Validate tolerance values."""
    if rtol < 100 * EPS:
        warn("`rtol` is too low, setting to {}".format(100 * EPS))
        rtol = 100 * EPS

    atol = xp.asarray(atol)
    if atol.ndim > 0 and atol.shape != (n,):
        raise ValueError("`atol` has wrong shape.")

    if xp.any(atol < 0):
        raise ValueError("`atol` must be positive.")

    return rtol, atol


def norm(x):
    """Compute RMS norm."""
    # this step looks like a bottleneck in some benchmarks.
    # This is because GPU and CPU have to sync in this function
    # Shape is portable host metadata (Torch.size is a method); only the
    # reduced norm crosses to the host for the adaptive step decision.
    return float(xp.linalg.norm(x)) / prod(x.shape) ** 0.5


def select_initial_step(fun, t0, y0, f0, direction, order, rtol, atol):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t0 : float
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    f0 : ndarray, shape (n,)
        Initial value of the derivative, i. e. ``fun(t0, y0)``.
    direction : float
        Integration direction.
    order : float
        Method order.
    rtol : float
        Desired relative tolerance.
    atol : float
        Desired absolute tolerance.

    Returns
    -------
    h_abs : float
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    # Shape is portable host metadata (see norm above).
    if prod(y0.shape) == 0:
        return xp.inf

    scale = atol + xp.abs(y0) * rtol
    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * direction * f0
    f1 = fun(t0 + h0 * direction, y1)
    d2 = norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))

    return min(100 * h0, h1)


class OdeSolution(object):
    """Continuous ODE solution.

    It is organized as a collection of `DenseOutput` objects which represent
    local interpolants. It provides an algorithm to select a right interpolant
    for each given point.

    The interpolants cover the range between `t_min` and `t_max` (see
    Attributes below). Evaluation outside this interval is not forbidden, but
    the accuracy is not guaranteed.

    When evaluating at a breakpoint (one of the values in `ts`) a segment with
    the lower index is selected.

    Parameters
    ----------
    ts : array_like, shape (n_segments + 1,)
        Time instants between which local interpolants are defined. Must
        be strictly increasing or decreasing (zero segment with two points is
        also allowed).
    interpolants : list of DenseOutput with n_segments elements
        Local interpolants. An i-th interpolant is assumed to be defined
        between ``ts[i]`` and ``ts[i + 1]``.

    Attributes
    ----------
    t_min, t_max : float
        Time range of the interpolation.
    """

    def __init__(self, ts, interpolants):
        # Evaluate later on the backend the solution was created with; times stay on the host.
        self._backend = xp.current
        ts = host_time_array(ts)
        d = np.diff(ts)
        # The first case covers integration on zero segment.
        if not ((ts.size == 2 and ts[0] == ts[-1]) or np.all(d > 0) or np.all(d < 0)):
            raise ValueError("`ts` must be strictly increasing or decreasing.")

        self.n_segments = len(interpolants)
        if ts.shape != (self.n_segments + 1,):
            raise ValueError("Numbers of time stamps and interpolants " "don't match.")

        self.ts = ts
        self.interpolants = interpolants
        if ts[-1] >= ts[0]:
            self.t_min = ts[0]
            self.t_max = ts[-1]
            self.ascending = True
            self.ts_sorted = ts
        else:
            self.t_min = ts[-1]
            self.t_max = ts[0]
            self.ascending = False
            self.ts_sorted = ts[::-1]

    def _call_single(self, t):
        # Here we preserve a certain symmetry that when t is in self.ts,
        # then we prioritize a segment with a lower index.
        if self.ascending:
            ind = np.searchsorted(self.ts_sorted, t, side="left")
        else:
            ind = np.searchsorted(self.ts_sorted, t, side="right")

        segment = min(max(ind - 1, 0), self.n_segments - 1)
        if not self.ascending:
            segment = self.n_segments - 1 - segment

        return self.interpolants[segment](t)

    def __call__(self, t):
        """Evaluate the solution.

        Parameters
        ----------
        t : float or array_like with shape (n_points,)
            Points to evaluate at.

        Returns
        -------
        y : ndarray, shape (n_states,) or (n_states, n_points)
            Computed values. Shape depends on whether `t` is a scalar or a
            1-d array.
        """
        from renormalizer.backend.context import capture_backend
        with capture_backend(self._backend):
            return self._call_impl(t)

    def _call_impl(self, t):
        t = host_time_array(t)

        if t.ndim == 0:
            return self._call_single(t)

        order = np.argsort(t)
        # argsort of a permutation is its inverse; same result as the indexed write.
        reverse = np.argsort(order)
        t_sorted = t[order]

        # See comment in self._call_single.
        if self.ascending:
            segments = np.searchsorted(self.ts_sorted, t_sorted, side="left")
        else:
            segments = np.searchsorted(self.ts_sorted, t_sorted, side="right")
        segments -= 1
        segments[segments < 0] = 0
        segments[segments > self.n_segments - 1] = self.n_segments - 1
        if not self.ascending:
            segments = self.n_segments - 1 - segments

        ys = []
        group_start = 0
        for segment, group in groupby(segments):
            group_end = group_start + len(list(group))
            y = self.interpolants[segment](t_sorted[group_start:group_end])
            ys.append(y)
            group_start = group_end

        ys = xp.hstack(ys)
        ys = ys[:, reverse]

        return ys


NUM_JAC_DIFF_REJECT = EPS ** 0.875
NUM_JAC_DIFF_SMALL = EPS ** 0.75
NUM_JAC_DIFF_BIG = EPS ** 0.25
NUM_JAC_MIN_FACTOR = 1e3 * EPS
NUM_JAC_FACTOR_INCREASE = 10
NUM_JAC_FACTOR_DECREASE = 0.1


def num_jac(fun, t, y, f, threshold, factor, sparsity=None):
    """Finite differences Jacobian approximation tailored for ODE solvers.

    This function computes finite difference approximation to the Jacobian
    matrix of `fun` with respect to `y` using forward differences.
    The Jacobian matrix has shape (n, n) and its element (i, j) is equal to
    ``d f_i / d y_j``.

    A special feature of this function is the ability to correct the step
    size from iteration to iteration. The main idea is to keep the finite
    difference significantly separated from its round-off error which
    approximately equals ``EPS * xp.abs(f)``. It reduces a possibility of a
    huge error and assures that the estimated derivative are reasonably close
    to the true values (i.e. the finite difference approximation is at least
    qualitatively reflects the structure of the true Jacobian).

    Parameters
    ----------
    fun : callable
        Right-hand side of the system implemented in a vectorized fashion.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    f : ndarray, shape (n,)
        Value of the right hand side at (t, y).
    threshold : float
        Threshold for `y` value used for computing the step size as
        ``factor * xp.maximum(xp.abs(y), threshold)``. Typically the value of
        absolute tolerance (atol) for a solver should be passed as `threshold`.
    factor : ndarray with shape (n,) or None
        Factor to use for computing the step size. Pass None for the very
        evaluation, then use the value returned from this function.
    sparsity : tuple (structure, groups) or None
        Sparsity structure of the Jacobian, `structure` must be csc_matrix.

    Returns
    -------
    J : ndarray or csc_matrix, shape (n, n)
        Jacobian matrix.
    factor : ndarray, shape (n,)
        Suggested `factor` for the next evaluation.
    """
    y = xp.asarray(y)
    n = y.shape[0]
    if n == 0:
        return xp.empty((0, 0)), factor

    if factor is None:
        factor = xp.full(n, EPS ** 0.5)
    else:
        # multibackend: copy through the adapter.
        factor = xp.array(factor, copy=True)

    # Direct the step as ODE dictates, hoping that such a step won't lead to
    # a problematic region. For complex ODEs it makes sense to use the real
    # part of f as we use steps along real axis.
    # multibackend: cast through the adapter in the backend's real dtype.
    f_sign = 2 * xp.asarray(xp.real(f) >= 0, dtype=xp.real_dtype) - 1
    y_scale = f_sign * xp.maximum(threshold, xp.abs(y))
    h = (y + factor * y_scale) - y

    # Make sure that the step is not 0 to start with. Not likely it will be
    # executed often.
    for i in xp.nonzero(h == 0)[0]:
        while h[i] == 0:
            # multibackend: writes go through the adapter; JAX arrays are immutable.
            factor = xp.write_owned(factor, i, factor[i] * 10)
            h = xp.write_owned(h, i, (y[i] + factor[i] * y_scale[i]) - y[i])

    if sparsity is None:
        return _dense_num_jac(fun, t, y, f, h, factor, y_scale)
    else:
        structure, groups = sparsity
        return _sparse_num_jac(fun, t, y, f, h, factor, y_scale, structure, groups)


def _dense_num_jac(fun, t, y, f, h, factor, y_scale):
    n = y.shape[0]
    h_vecs = xp.diag(h)
    f_new = fun(t, y[:, None] + h_vecs)
    diff = f_new - f[:, None]
    max_ind = xp.argmax(xp.abs(diff), axis=0)
    r = xp.arange(n)
    max_diff = xp.abs(diff[max_ind, r])
    scale = xp.maximum(xp.abs(f[max_ind]), xp.abs(f_new[max_ind, r]))

    diff_too_small = max_diff < NUM_JAC_DIFF_REJECT * scale
    if xp.any(diff_too_small):
        ind, = xp.nonzero(diff_too_small)
        new_factor = NUM_JAC_FACTOR_INCREASE * factor[ind]
        h_new = (y[ind] + new_factor * y_scale[ind]) - y[ind]
        # multibackend: writes go through the adapter; JAX arrays are immutable.
        h_vecs = xp.write_owned(h_vecs, (ind, ind), h_new)
        f_new = fun(t, y[:, None] + h_vecs[:, ind])
        diff_new = f_new - f[:, None]
        max_ind = xp.argmax(xp.abs(diff_new), axis=0)
        r = xp.arange(ind.shape[0])
        max_diff_new = xp.abs(diff_new[max_ind, r])
        scale_new = xp.maximum(xp.abs(f[max_ind]), xp.abs(f_new[max_ind, r]))

        update = max_diff[ind] * scale_new < max_diff_new * scale[ind]
        if xp.any(update):
            update, = xp.nonzero(update)
            update_ind = ind[update]
            # multibackend: as above.
            factor = xp.write_owned(factor, update_ind, new_factor[update])
            h = xp.write_owned(h, update_ind, h_new[update])
            diff = xp.write_owned(diff, (slice(None), update_ind), diff_new[:, update])
            scale = xp.write_owned(scale, update_ind, scale_new[update])
            max_diff = xp.write_owned(max_diff, update_ind, max_diff_new[update])

    # Out of place: JAX arrays are immutable.
    diff = diff / h

    # xp.where instead of masked in-place multiplication; same values.
    factor = xp.where(max_diff < NUM_JAC_DIFF_SMALL * scale, factor * NUM_JAC_FACTOR_INCREASE, factor)
    factor = xp.where(max_diff > NUM_JAC_DIFF_BIG * scale, factor * NUM_JAC_FACTOR_DECREASE, factor)
    factor = xp.maximum(factor, NUM_JAC_MIN_FACTOR)

    return diff, factor


def _sparse_num_jac(fun, t, y, f, h, factor, y_scale, structure, groups):
    # SciPy owns sparse storage and sparsity metadata. Only gathered derivative
    # entries cross that boundary; perturbations and RHS evaluation stay native.
    n = y.shape[0]
    groups = np.asarray(xp.numpy(groups) if xp.current.is_array(groups) else groups, dtype=int)
    n_groups = int(np.max(groups)) + 1
    h_vecs = xp.stack([h * xp.asarray(groups == group) for group in range(n_groups)]).T
    f_new = fun(t, y[:, None] + h_vecs)
    df = f_new - f[:, None]
    i, j, _ = find(structure)
    entries = xp.numpy(df[xp.asarray(i), xp.asarray(groups[j])])
    diff = coo_matrix((entries, (i, j)), shape=(n, n)).tocsc()
    max_ind_host = np.asarray(abs(diff).argmax(axis=0)).ravel()
    max_ind = xp.asarray(max_ind_host)
    r = xp.arange(n)
    max_diff = xp.asarray(np.asarray(abs(diff[max_ind_host, np.arange(n)])).ravel())
    scale = xp.maximum(xp.abs(f[max_ind]), xp.abs(f_new[max_ind, xp.asarray(groups)]))

    diff_too_small = max_diff < NUM_JAC_DIFF_REJECT * scale
    if xp.any(diff_too_small):
        ind, = xp.nonzero(diff_too_small)
        ind_host = np.asarray(xp.numpy(ind))
        new_factor = NUM_JAC_FACTOR_INCREASE * factor[ind]
        h_new = (y[ind] + new_factor * y_scale[ind]) - y[ind]
        h_new_all = xp.zeros(n, dtype=h.dtype)
        h_new_all = xp.write_owned(h_new_all, ind, h_new)
        groups_unique = np.unique(groups[ind_host])
        groups_map = np.empty(n_groups, dtype=int)
        for k, group in enumerate(groups_unique):
            groups_map[group] = k
        h_vecs = xp.stack([h_new_all * xp.asarray(groups == group) for group in groups_unique]).T
        f_new = fun(t, y[:, None] + h_vecs)
        df = f_new - f[:, None]
        i, j, _ = find(structure[:, ind_host])
        entries = xp.numpy(df[xp.asarray(i), xp.asarray(groups_map[groups[ind_host[j]]])])
        diff_new = coo_matrix((entries, (i, j)), shape=(n, len(ind_host))).tocsc()
        max_ind_new_host = np.asarray(abs(diff_new).argmax(axis=0)).ravel()
        max_ind_new = xp.asarray(max_ind_new_host)
        max_diff_new = xp.asarray(np.asarray(abs(diff_new[max_ind_new_host, np.arange(len(ind_host))])).ravel())
        scale_new = xp.maximum(xp.abs(f[max_ind_new]),
                               xp.abs(f_new[max_ind_new, xp.asarray(groups_map[groups[ind_host]])]))
        update = max_diff[ind] * scale_new < max_diff_new * scale[ind]
        if xp.any(update):
            update, = xp.nonzero(update)
            update_ind = ind[update]
            # multibackend: as above.
            factor = xp.write_owned(factor, update_ind, new_factor[update])
            h = xp.write_owned(h, update_ind, h_new[update])
            diff[:, np.asarray(xp.numpy(update_ind))] = diff_new[:, np.asarray(xp.numpy(update))]
            scale = xp.write_owned(scale, update_ind, scale_new[update])
            max_diff = xp.write_owned(max_diff, update_ind, max_diff_new[update])

    # diff is SciPy sparse (host) storage; h is brought to the host for it.
    diff.data /= np.repeat(np.asarray(xp.numpy(h)), np.diff(diff.indptr))
    factor = xp.where(max_diff < NUM_JAC_DIFF_SMALL * scale, factor * NUM_JAC_FACTOR_INCREASE, factor)
    factor = xp.where(max_diff > NUM_JAC_DIFF_BIG * scale, factor * NUM_JAC_FACTOR_DECREASE, factor)
    return diff, xp.maximum(factor, NUM_JAC_MIN_FACTOR)
