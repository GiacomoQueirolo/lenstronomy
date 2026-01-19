__author__ = "giacomoqueirolo"

import numpy as np

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
__all__ = ["ParallelArsinh"]


"""
Fails due to race-conditions
@njit(parallel=True, fastmath=False)
def _derivatives_numba(x, y, theta_E, theta_c, center_x, center_y):
    print("using derivatives numba2")
    # if x,y is 2D, must be flattened
    flat = True
    if x.ndim == 2:
        shp = x.shape
        x_flat = x.ravel()
        y_flat = y.ravel()
        flat = False
    else:
        x_flat = x
        y_flat = y
    n_pix   = x_flat.size
    n_lens  = theta_E.size
    alpha_x = np.zeros(n_pix)
    alpha_y = np.zeros(n_pix)

    for j in prange(n_pix):  # parallel over pixels
        ax = 0.0
        ay = 0.0
        xj = x_flat[j]
        yj = y_flat[j]
        for i in range(n_lens):
            dx = xj - center_x[i]
            dy = yj - center_y[i]
            r2 = dx * dx + dy * dy
            te2 = theta_E[i] * theta_E[i]
            tc4 = theta_c[i] * theta_c[i] * theta_c[i] * theta_c[i]
            f = te2 / np.sqrt(tc4 + r2 * r2)
            ax += f * dx
            ay += f * dy
        alpha_x[j] = ax
        alpha_y[j] = ay
    if not flat:
        alpha_x = alpha_x.reshape(shp)
        alpha_y = alpha_y.reshape(shp)
    return alpha_x, alpha_y
@njit(parallel=True, fastmath=True)
def _derivatives_numba(x, y, theta_E, theta_c, center_x, center_y, chunk_size=1000):
    print("Chunked version")
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

    alpha_x = np.zeros(n_pix)
    alpha_y = np.zeros(n_pix)

    for start in range(0, n_lens, chunk_size):
        end = min(start + chunk_size, n_lens)
        tmp_x = np.zeros(n_pix)
        tmp_y = np.zeros(n_pix)

        for i in prange(start, end):
            tE2 = theta_E[i] * theta_E[i]
            tc4 = theta_c[i] * theta_c[i] * theta_c[i] * theta_c[i]
            cx = center_x[i]
            cy = center_y[i]
            for j in range(n_pix):
                dx = x_flat[j] - cx
                dy = y_flat[j] - cy
                r2 = dx * dx + dy * dy
                f = tE2 / np.sqrt(tc4 + r2 * r2)
                tmp_x[j] += f * dx
                tmp_y[j] += f * dy

        alpha_x += tmp_x
        alpha_y += tmp_y

    if not flat:
        alpha_x = alpha_x.reshape(shp)
        alpha_y = alpha_y.reshape(shp)

    return alpha_x, alpha_y
"""
import numpy as np
from numba import njit, prange

# Helper to pick chunk size (num lenses per chunk) given memory budget.
# Each chunk allocates two arrays: tmp_x, tmp_y of shape (chunk_len, n_pix) float64.
def choose_chunk_size(n_pix, max_mem_bytes=200*1024**2):
    # bytes per element (float64) = 8. two arrays => factor 2.
    bytes_per_element = 8 * 2
    # chunk_len * n_pix * bytes_per_element <= max_mem_bytes
    chunk_len = max(int(max_mem_bytes // (n_pix * bytes_per_element)), 1)
    return chunk_len


@njit(parallel=False, fastmath=False)
def _derivatives_numba_chunked_lens_parallel(x, y,
                                             theta_E, theta_c,
                                             center_x, center_y,
                                             chunk_len):
    """
    Chunked, lens-parallel, thread-safe derivatives.
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
    
@njit(parallel=False, fastmath=False)
def _hessian_numba_chunked_lens_parallel(x, y,
                                             theta_E, theta_c,
                                             center_x, center_y,
                                             chunk_len):
    """
    Chunked, lens-parallel, thread-safe hessian
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

    n_pix  = x_flat.size
    n_lens = theta_E.size

    # global accumulators (1D, length n_pix)
    f_xx = np.zeros(n_pix, dtype=np.float64)
    f_yy = np.zeros(n_pix, dtype=np.float64)
    f_xy = np.zeros(n_pix, dtype=np.float64)
    
    # loop over lens chunks (serial, to limit memory)
    start = 0
    while start < n_lens:
        end = start + chunk_len
        if end > n_lens:
            end = n_lens
        cur_len = end - start

        # Allocate per-lens rows for this chunk (cur_len x n_pix).
        # Each lens i in [start,end) writes only to row (i-start).
        tmp_xx = np.zeros((cur_len, n_pix), dtype=np.float64)
        tmp_yy = np.zeros((cur_len, n_pix), dtype=np.float64)
        tmp_xy = np.zeros((cur_len, n_pix), dtype=np.float64)
        
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

            row_xx = tmp_xx[ii]  # view to row ii (lens=ii)
            row_yy = tmp_yy[ii]
            row_xy = tmp_xy[ii]

            # inner loop over pixels (serial per lens)
            for j in range(n_pix):
                dx = x_flat[j] - cx
                dy = y_flat[j] - cy
                r2 = dx * dx + dy * dy
                denom = (tc4 + r2 * r2)**(1.5)
                l4 = te2 / denom
                kappa = l4 * tc4
                gamma1 = - l4 * r2 * (dx*dx - dy*dy)
                gamma2 = - l4 * r2 * 2 * dx * dy

                row_xx[j] = kappa + gamma1
                row_yy[j] = kappa - gamma1
                row_xy[j] = gamma2 

        # Now reduce (sum rows) - deterministic serial reduction
        # Sum rows into chunk_sum_ij then add to global f_ij
        chunk_sum_xx = np.zeros(n_pix, dtype=np.float64)
        chunk_sum_yy = np.zeros(n_pix, dtype=np.float64)
        chunk_sum_xy = np.zeros(n_pix, dtype=np.float64)
        # summation order is deterministic (row-major)
        for ii in range(cur_len):
            row_xx = tmp_xx[ii]
            row_yy = tmp_yy[ii]
            row_xy = tmp_xy[ii]
            for j in range(n_pix):
                chunk_sum_xx[j] += row_xx[j]
                chunk_sum_yy[j] += row_yy[j]
                chunk_sum_xy[j] += row_xy[j]

        # accumulate chunk contribution
        for j in range(n_pix):
            f_xx[j] += chunk_sum_xx[j]
            f_xy[j] += chunk_sum_xy[j]
            f_yy[j] += chunk_sum_yy[j]

        # advance
        start = end

    # restore shape if needed
    if not flat:
        f_xx = f_xx.reshape(shp)
        f_yy = f_yy.reshape(shp)
        f_xy = f_xy.reshape(shp)

    return f_xx, f_xy, f_xy, f_yy

class ParallelArsinh(LensProfileBase):
    """
    Exactly as Arsinh, but now takes as input all the particle parameters and does the computation vectorially
    The arsinh lens is a fully analytical lens model designed to regularise the point lens.
    The model behaves as a point lens far from the centre, while having a finite density at the centre.
    The name comes from its lensing potential, which involves the arsinh function.
    
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
        "theta_c": 1e-6,
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

    def _dxdy(self,x,y,center_x,center_y):
        x        = np.asarray(x, dtype=np.float64)
        y        = np.asarray(y, dtype=np.float64)
     
        center_x = np.atleast_1d(np.asarray(center_x, dtype=np.float64))
        center_y = np.atleast_1d(np.asarray(center_y, dtype=np.float64))
        dx       = x[np.newaxis, :] - center_x[:, np.newaxis] # shape N_x,M_cnt
        dy       = y[np.newaxis, :] - center_y[:, np.newaxis]  
        """
        center_x = np.asarray(center_x, dtype=np.float64)
        center_y = np.asarray(center_y, dtype=np.float64)
        if center_x.ndim==0 or x.ndim==0:
            # either one or both are floats:
            dx = x - center_x
            dy = y - center_y
        elif center_x.ndim==1 and x.ndim==1: # both should be 1D vectors,but w different N
            dx = x[np.newaxis, :] - center_x[:, np.newaxis] # shape N_x,M_cnt
            dy = y[np.newaxis, :] - center_y[:, np.newaxis]  
        """
        return dx,dy

    def function(self, x, y, theta_E, theta_c, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :param theta_c: core radius (in angles)
        :return: lensing potential (in squared angles)
        """
        x_,y_ = self._dxdy(x,y,center_x,center_y)
        r2    = x_*x_ + y_*y_

        theta_E = np.atleast_1d(np.asarray(theta_E, dtype=np.float64))[:,np.newaxis]
        
        psi = 0.5 * theta_E**2 * np.arcsinh(r2 / theta_c**2)
        # we have to sum over all particles
        psi = psi.sum(axis=0)
        return psi


    def derivatives(self, x, y, theta_E, theta_c, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :param theta_c: core radius (in angles)
        :return: displacement angle (in angles)
        """
        if len(theta_E)>1e3:
            #return _derivatives_numba(x, y, theta_E, theta_c, center_x, center_y)
            chunk_len = choose_chunk_size(x.size)
            return _derivatives_numba_chunked_lens_parallel(x, y, theta_E, theta_c, center_x, center_y,chunk_len=chunk_len)
        x_,y_ = self._dxdy(x,y,center_x,center_y)
        r2    = x_*x_ + y_*y_
        theta_E = np.atleast_1d(np.asarray(theta_E, dtype=np.float64))[:,np.newaxis]
        theta_c = np.atleast_1d(np.asarray(theta_c, dtype=np.float64))[:,np.newaxis]

        f = theta_E*theta_E / np.sqrt(theta_c*theta_c*theta_c*theta_c + r2*r2)
        
        # we have to sum over all particles
        alpha_x = np.sum(f * x_, axis=0)
        alpha_y = np.sum(f * y_, axis=0)

        return alpha_x, alpha_y

    def hessian(self, x, y, theta_E, theta_c, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :param theta_c: core radius (in angles)
        :return: Hessian matrix (dimensionless)
        """
        if len(theta_E)>1e3:
            #return _derivatives_numba(x, y, theta_E, theta_c, center_x, center_y)
            chunk_len = choose_chunk_size(x.size)
            return _hessian_numba_chunked_lens_parallel(x, y, theta_E, theta_c, center_x, center_y,chunk_len=chunk_len)
        x_,y_ = self._dxdy(x,y,center_x,center_y)
        r2    = x_*x_ + y_*y_
        theta_E = np.atleast_1d(np.asarray(theta_E, dtype=np.float64))[:,np.newaxis]
        theta_c = np.atleast_1d(np.asarray(theta_c, dtype=np.float64))[:,np.newaxis]

        tc4 = theta_c*theta_c*theta_c*theta_c
        l4  = theta_E*theta_E / (tc4 + r2*r2)**(1.5)
        kappa = l4 * tc4
        gamma1 = - l4 * r2 * (x_*x_ - y_*y_)
        gamma2 = - l4 * r2 * 2 * x_ * y_
        
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        
        if f_xx.ndim==2:
            # we have to sum over all particles
            f_xx = f_xx.sum(axis=0)
            f_yy = f_yy.sum(axis=0)
            f_xy = f_xy.sum(axis=0)
        return f_xx, f_xy, f_xy, f_yy
