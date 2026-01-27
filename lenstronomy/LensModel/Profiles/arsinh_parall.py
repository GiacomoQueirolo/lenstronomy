__author__ = "giacomoqueirolo"

import numpy as np
from functools import wraps
from numba import njit, prange

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["ParallelArsinh"]


def enforce_bounds():
    """High-performance bounds checker."""
    param_names = "theta_E", "theta_c", "center_x", "center_y"

    def decorator(func):
        # resolve parameter positions ONCE
        code = func.__code__
        arg_names = code.co_varnames[: code.co_argcount]

        indices = []
        for name in param_names:
            if name not in arg_names:
                raise ValueError(f"{name} not in {func.__name__} signature")
            indices.append(arg_names.index(name))

        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]

            lower = self.lower_limit_default
            upper = self.upper_limit_default

            for idx, name in zip(indices, param_names):
                if idx < len(args):
                    v = args[idx]
                else:
                    v = kwargs.get(name)

                if v is None:
                    continue

                # Fast scalar path
                if np.isscalar(v):
                    if v < lower[name] or v > upper[name]:
                        raise ValueError(
                            f"{func.__name__}: {name}={v} "
                            f"outside [{lower[name]}, {upper[name]}]"
                        )
                    continue

                # Array path
                arr = np.asarray(v)
                vmin = arr.min()
                vmax = arr.max()

                if vmin < lower[name] or vmax > upper[name]:
                    raise ValueError(
                        f"{func.__name__}: {name} outside bounds "
                        f"[{lower[name]}, {upper[name]}], "
                        f"got [{vmin}, {vmax}]"
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def choose_chunk_size(n_pix, max_mem_bytes=200 * 1024**2):
    """Helper to pick chunk size (num lenses per chunk) given memory budget.

    Each chunk allocates two arrays: tmp_x, tmp_y of shape (chunk_len, n_pix) float64.
    """
    # bytes per element (float64) = 8. two arrays => factor 2.
    bytes_per_element = 8 * 2
    # chunk_len * n_pix * bytes_per_element <= max_mem_bytes
    chunk_len = max(int(max_mem_bytes // (n_pix * bytes_per_element)), 1)
    return chunk_len


@njit(parallel=True, fastmath=False)
def _function_numba_chunked_lens_parallel(
    x, y, theta_E, theta_c, center_x, center_y, chunk_len
):
    """Chunked, lens-parallel, thread-safe function (potential).

    - Parallelization is over lenses inside each chunk (prange).
    - Each lens writes to its own row in tmp arrays (no race).
    - After the chunk's prange, rows are summed and added to potential map.
    Note: >> chunk_len, >> memory intensive, >> speed
    """
    # Flatten inputs if 2D (we keep shape info to restore later)
    if x.ndim == 2:
        shp = x.shape
        x_flat = x.ravel()
        y_flat = y.ravel()
        flat = False
    else:
        x_flat = x
        y_flat = y
        flat = True

    n_pix = x_flat.size
    n_lens = theta_E.size

    # global accumulator (1D, length n_pix)
    psi = np.zeros(n_pix, dtype=np.float64)

    # loop over lens chunks (serial, to limit memory)
    start = 0
    while start < n_lens:
        end = start + chunk_len
        if end > n_lens:
            end = n_lens
        cur_len = end - start

        # Allocate per-lens rows for this chunk (cur_len x n_pix).
        # Each lens i in [start,end) writes only to row (i-start).
        tmp = np.zeros((cur_len, n_pix), dtype=np.float64)

        # Parallel loop over lenses in this chunk.
        # Each i writes into tmp [i-start, j] only.
        for ii in prange(cur_len):
            i = start + ii
            te = theta_E[i]
            tc = theta_c[i]
            te2 = te * te
            tc2 = tc * tc
            cx = center_x[i]
            cy = center_y[i]

            row = tmp[ii]  # view to row ii (lens=ii)

            # inner loop over pixels (serial per lens)
            for j in range(n_pix):
                dx = x_flat[j] - cx
                dy = y_flat[j] - cy
                r2 = dx * dx + dy * dy
                arsh = np.arcsinh(r2 / tc2)
                f = 0.5 * te2 * arsh
                row[j] = f
        # Now reduce (sum rows) - deterministic serial reduction
        # Sum rows into chunk_sum then add to global potential
        chunk_sum = np.zeros(n_pix, dtype=np.float64)
        # summation order is deterministic (row-major)
        for ii in range(cur_len):
            row = tmp[ii]
            for j in range(n_pix):
                chunk_sum[j] += row[j]

        # accumulate chunk contribution
        for j in range(n_pix):
            psi[j] += chunk_sum[j]

        # advance
        start = end

    # restore shape if needed
    if not flat:
        psi = psi.reshape(shp)

    return psi


@njit(parallel=True, fastmath=False)
def _derivatives_numba_chunked_lens_parallel(
    x, y, theta_E, theta_c, center_x, center_y, chunk_len
):
    """Chunked, lens-parallel, thread-safe derivatives.

    - Parallelization is over lenses inside each chunk (prange).
    - Each lens writes to its own row in tmp arrays (no race).
    - After the chunk's prange, rows are summed and added to alpha arrays.
    Note: >> chunk_len, >> memory intensive, >> speed
    """
    # Flatten inputs if 2D (we keep shape info to restore later)
    if x.ndim == 2:
        shp = x.shape
        x_flat = x.ravel()
        y_flat = y.ravel()
        flat = False
    else:
        x_flat = x
        y_flat = y
        flat = True

    n_pix = x_flat.size
    n_lens = theta_E.size

    # global accumulators (1D, length n_pix)
    alpha_x = np.zeros(n_pix, dtype=np.float64)
    alpha_y = np.zeros(n_pix, dtype=np.float64)

    # loop over lens chunks (serial, to limit memory)
    start = 0
    while start < n_lens:
        end = start + chunk_len
        if end > n_lens:
            end = n_lens
        cur_len = end - start

        # Allocate per-lens rows for this chunk (cur_len x n_pix).
        # Each lens i in [start,end) writes only to row (i-start).
        tmp_x = np.zeros((cur_len, n_pix), dtype=np.float64)
        tmp_y = np.zeros((cur_len, n_pix), dtype=np.float64)

        # Parallel loop over lenses in this chunk.
        # Each i writes into tmp_x[i-start, j] and tmp_y[i-start, j] only.
        for ii in prange(cur_len):
            i = start + ii
            te = theta_E[i]
            tc = theta_c[i]
            te2 = te * te
            tc4 = tc * tc * tc * tc
            cx = center_x[i]
            cy = center_y[i]

            row_x = tmp_x[ii]  # view to row ii (lens=ii)
            row_y = tmp_y[ii]

            # inner loop over pixels (serial per lens)
            for j in range(n_pix):
                dx = x_flat[j] - cx
                dy = y_flat[j] - cy
                r2 = dx * dx + dy * dy
                denom = np.sqrt(tc4 + r2 * r2)
                f = te2 / denom
                row_x[j] = f * dx
                row_y[j] = f * dy

        # Now reduce (sum rows) - deterministic serial reduction
        # Sum rows into chunk_sum_x / chunk_sum_y then add to global alpha
        chunk_sum_x = np.zeros(n_pix, dtype=np.float64)
        chunk_sum_y = np.zeros(n_pix, dtype=np.float64)
        # summation order is deterministic (row-major)
        for ii in range(cur_len):
            row_x = tmp_x[ii]
            row_y = tmp_y[ii]
            for j in range(n_pix):
                chunk_sum_x[j] += row_x[j]
                chunk_sum_y[j] += row_y[j]

        # accumulate chunk contribution
        for j in range(n_pix):
            alpha_x[j] += chunk_sum_x[j]
            alpha_y[j] += chunk_sum_y[j]

        # advance
        start = end

    # restore shape if needed
    if not flat:
        alpha_x = alpha_x.reshape(shp)
        alpha_y = alpha_y.reshape(shp)

    return alpha_x, alpha_y


@njit(parallel=True, fastmath=False)
def _hessian_numba_chunked_lens_parallel(
    x, y, theta_E, theta_c, center_x, center_y, chunk_len
):
    # Flatten inputs if 2D
    if x.ndim == 2:
        shp = x.shape
        x_flat = x.ravel()
        y_flat = y.ravel()
        reshape_out = True
    else:
        x_flat = x
        y_flat = y
        shp = None
        reshape_out = False

    n_pix = x_flat.size
    n_lens = theta_E.size

    # final outputs
    f_xx = np.zeros(n_pix, np.float64)
    f_yy = np.zeros(n_pix, np.float64)
    f_xy = np.zeros(n_pix, np.float64)

    start = 0
    while start < n_lens:
        end = start + chunk_len
        if end > n_lens:
            end = n_lens
        cur_len = end - start

        # each thread gets a private buffer to accumulate into
        buf_xx = np.zeros((cur_len, n_pix), np.float64)
        buf_yy = np.zeros((cur_len, n_pix), np.float64)
        buf_xy = np.zeros((cur_len, n_pix), np.float64)

        # parallel loop over lenses in the chunk
        for ii in prange(cur_len):
            i = start + ii
            te = theta_E[i]
            tc = theta_c[i]
            cx = center_x[i]
            cy = center_y[i]

            te2 = te * te
            tc4 = tc * tc * tc * tc

            row_xx = buf_xx[ii]
            row_yy = buf_yy[ii]
            row_xy = buf_xy[ii]

            for j in range(n_pix):
                dx = x_flat[j] - cx
                dy = y_flat[j] - cy

                r2 = dx * dx + dy * dy
                den = tc4 + r2 * r2
                den = den * np.sqrt(den)  # faster than **1.5

                l4 = te2 / den
                kappa = l4 * tc4
                g1 = -l4 * r2 * (dx * dx - dy * dy)
                g2 = -l4 * r2 * (2.0 * dx * dy)

                row_xx[j] = kappa + g1
                row_yy[j] = kappa - g1
                row_xy[j] = g2

        # serial reduction of per-thread buffers
        for ii in range(cur_len):
            row_xx = buf_xx[ii]
            row_yy = buf_yy[ii]
            row_xy = buf_xy[ii]
            for j in range(n_pix):
                f_xx[j] += row_xx[j]
                f_yy[j] += row_yy[j]
                f_xy[j] += row_xy[j]

        start = end

    if reshape_out:
        return (
            f_xx.reshape(shp),
            f_xy.reshape(shp),
            f_xy.reshape(shp),
            f_yy.reshape(shp),
        )
    else:
        return f_xx, f_xy, f_xy, f_yy


def _dxdy(x, y, center_x, center_y):
    """Simple optimised function to compute the distances from the centre.

    :param x: x-coord (in angles)
    :param y: y-coord (in angles)
    :param center_x: center x-coord (in angles)
    :param center_y: center y-coord (in angles)
    :return: dx, dy distances from the center (in angles)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    center_x = np.atleast_1d(np.asarray(center_x, dtype=np.float64))
    center_y = np.atleast_1d(np.asarray(center_y, dtype=np.float64))
    dx = x[np.newaxis, :] - center_x[:, np.newaxis]  # shape N_x,M_cnt
    dy = y[np.newaxis, :] - center_y[:, np.newaxis]
    return dx, dy


class ParallelArsinh(LensProfileBase):
    """Exactly as Arsinh, but now takes as input all the particle parameters and does
    the computation vectorially. The arsinh lens is a fully analytical lens model
    designed to regularise the point lens. The model behaves as a point lens far from
    the centre, while having a finite density at the centre. The name comes from its
    lensing potential, which involves the arsinh function.

       The lensing potential, displacement angle, convergence and shear read

       .. math::
           \\psi(\\theta)
    = \\frac{\\theta_{\\rm E}^2}{2}
           \\mathrm{arsinh}\\left(\\frac{\\theta}{\\theta_{\\rm c}}\\right)^2

       .. math::
           \\boldsymbol{\\alpha}(\\boldsymbol{\\theta})
    = \\frac{\\theta_{\\rm E}^2\\boldsymbol{\\theta}}
                   {\\sqrt{\\theta_{\\rm c}^4+\\theta^4}}

       .. math::
           \\kappa(\\theta)
    = \\frac{\\theta_{\\rm E}^2\\theta_{\\rm c}^4}
                   {(\\theta_{\\rm c}^4+\\theta^4)^{3/2}}

       .. math::
           \\gamma(\\boldsymbol{\\theta})
    = - \\frac{\\theta_{\\rm E}^2\\theta^2}
                     {(\\theta_{\\rm c}^4+\\theta^4)^{3/2}}
            (\\theta_1 + \\mathrm{i}\\theta_2)^2

       with :math:`\\theta_{\\rm E}` the Einstein radius and :math:`\\theta_{\\rm c}` the core radius of the lens.

       The central convergence is finite, :math:`\\kappa(0)=(\\theta_{\\rm E}/\\theta_{\\rm c})^2`, as well as the total
       mass. The mass enclosed in a disk o

       .. math::
           M = \\frac{\\theta_{\\rm E}^2 D_{\\rm d} D_{\\rm s}}{4G D_{\\rm ds}}

       This model was used for illustrations in Sec. 4 of Fleury et al. (2021) https://arxiv.org/abs/2104.08883.
    """

    param_names = ["theta_E", "theta_c", "center_x", "center_y"]
    lower_limit_default = {
        "theta_E": 0,
        "theta_c": 1e-12,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "theta_E": 100,
        "theta_c": 100,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self):
        super(ParallelArsinh, self).__init__()

    @enforce_bounds()
    def function(self, x, y, theta_E, theta_c, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :param theta_c: core radius (in angles)
        :return: lensing potential (in squared angles)
        """
        if len(theta_E) > 1e3:
            # parallelised version for large particle number
            chunk_len = choose_chunk_size(x.size)
            return _function_numba_chunked_lens_parallel(
                x, y, theta_E, theta_c, center_x, center_y, chunk_len=chunk_len
            )

        x_, y_ = _dxdy(x, y, center_x, center_y)
        r2 = x_ * x_ + y_ * y_

        theta_E = np.atleast_1d(np.asarray(theta_E, dtype=np.float64))[:, np.newaxis]
        theta_c = np.atleast_1d(np.asarray(theta_c, dtype=np.float64))[:, np.newaxis]

        psi = 0.5 * theta_E * theta_E * np.arcsinh(r2 / (theta_c * theta_c))
        # Sum over all particles
        psi = psi.sum(axis=0)
        return psi

    @enforce_bounds()
    def derivatives(self, x, y, theta_E, theta_c, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :param theta_c: core radius (in angles)
        :return: displacement angle (in angles)
        """
        if len(theta_E) > 1e3:
            # parallelised version for large particle number
            chunk_len = choose_chunk_size(x.size)
            return _derivatives_numba_chunked_lens_parallel(
                x, y, theta_E, theta_c, center_x, center_y, chunk_len=chunk_len
            )
        x_, y_ = _dxdy(x, y, center_x, center_y)
        r2 = x_ * x_ + y_ * y_
        theta_E = np.atleast_1d(np.asarray(theta_E, dtype=np.float64))[:, np.newaxis]
        theta_c = np.atleast_1d(np.asarray(theta_c, dtype=np.float64))[:, np.newaxis]

        f = theta_E * theta_E / np.sqrt(theta_c * theta_c * theta_c * theta_c + r2 * r2)

        # Sum over all particles
        alpha_x = np.sum(f * x_, axis=0)
        alpha_y = np.sum(f * y_, axis=0)

        return alpha_x, alpha_y

    @enforce_bounds()
    def hessian(self, x, y, theta_E, theta_c, center_x=0, center_y=0):
        """
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :param theta_c: core radius (in angles)
        :return: Hessian matrix (dimensionless)
        """
        if len(theta_E) > 1e3:
            # parallelised version for large particle number
            chunk_len = choose_chunk_size(x.size)
            return _hessian_numba_chunked_lens_parallel(
                x, y, theta_E, theta_c, center_x, center_y, chunk_len=chunk_len
            )
        x_, y_ = _dxdy(x, y, center_x, center_y)
        r2 = x_ * x_ + y_ * y_
        theta_E = np.atleast_1d(np.asarray(theta_E, dtype=np.float64))[:, np.newaxis]
        theta_c = np.atleast_1d(np.asarray(theta_c, dtype=np.float64))[:, np.newaxis]

        tc4 = theta_c * theta_c * theta_c * theta_c
        l4 = theta_E * theta_E / (tc4 + r2 * r2) ** (1.5)
        kappa = l4 * tc4
        gamma1 = -l4 * r2 * (x_ * x_ - y_ * y_)
        gamma2 = -l4 * r2 * 2 * x_ * y_

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2

        if f_xx.ndim == 2:
            # Sum over all particles
            f_xx = f_xx.sum(axis=0)
            f_yy = f_yy.sum(axis=0)
            f_xy = f_xy.sum(axis=0)
        return f_xx, f_xy, f_xy, f_yy
