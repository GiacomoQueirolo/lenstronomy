__author__ = 'giacomo queirolo'


import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['ParallelPointMass']

from numba import njit, prange

# Helper to pick chunk size (num lenses per chunk) given memory budget.
# Each chunk allocates two arrays: tmp_x, tmp_y of shape (chunk_len, n_pix) float64.
def choose_chunk_size(n_pix, max_mem_bytes=200*1024**2): # Same as arshin_parall
    # bytes per element (float64) = 8. two arrays => factor 2.
    bytes_per_element = 8 * 2
    # chunk_len * n_pix * bytes_per_element <= max_mem_bytes
    chunk_len = max(int(max_mem_bytes // (n_pix * bytes_per_element)), 1)
    return chunk_len



@njit
def clamp_min_inplace(a, rmin):
    flat = a.ravel()
    for i in range(flat.size):
        if flat[i] < rmin:
            flat[i] = rmin

             
@njit(parallel=False, fastmath=False)
def _derivatives_numba_chunked_lens_parallel(x, y,
                                             theta_E,r_min,
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
            i   = start + ii
            te  = theta_E[i]
            te2 = te * te

            cx = center_x[i]
            cy = center_y[i]

            row_x = tmp_x[ii]  # view to row ii (lens=ii)
            row_y = tmp_y[ii]

            # inner loop over pixels (serial per lens)
            for j in range(n_pix):
                dx = x_flat[j] - cx
                dy = y_flat[j] - cy
                r2 = dx * dx + dy * dy
                #r2 = clamp_min(r2,(r_min**2))
                #clamp_min_inplace_scalar(r2,r_min*r_min)
                r2 = np.maximum(r2, r_min**2)
                f  = te2 / r2
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
                                             theta_E,
                                             center_x, center_y,
                                             chunk_len):
    """
    Chunked, lens-parallel, thread-safe hessias.
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
            te2 = te * te
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
                r4 = r2*r2
                l4 = te2 / r4
                _xx    = l4 *(dy*dy- dx*dx)
                _yy    = l4 *(dx*dx- dy*dy)
                gamma2 = - l4 * 2 * dx * dy

                row_xx[j] = _xx
                row_yy[j] = _yy
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


class ParallelPointMass(LensProfileBase):
    """
    Exactly as PointMass, but now takes as input all the particle parameters and does the computation vectorially
    class to compute the physical deflection angle of a point mass, given as an Einstein radius
    """
    param_names = ['theta_E', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 100, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.r_min = 10**(-25)
        super(ParallelPointMass, self).__init__()
        # alpha = 4*const.G * (mass*const.M_sun)/const.c**2/(r*const.Mpc)
        
    def _dxdy(self,x,y,center_x,center_y):
        x        = np.asarray(x, dtype=np.float64)
        y        = np.asarray(y, dtype=np.float64)
     
        center_x = np.atleast_1d(np.asarray(center_x, dtype=np.float64))
        center_y = np.atleast_1d(np.asarray(center_y, dtype=np.float64))
        dx       = x[np.newaxis, :] - center_x[:, np.newaxis] # shape N_x,M_cnt
        dy       = y[np.newaxis, :] - center_y[:, np.newaxis]  
        return dx,dy
    def _r(self,dx,dy,center_x,center_y):
        r     = np.sqrt(dx**2 + dy**2)
        #a    = np.sqrt(x_**2 + y_**2)
        #r     = np.empty_like(a)
        #r[a > self.r_min]  = a[a > self.r_min]  
        #r[a <= self.r_min] = self.r_min
        clamp_min_inplace(r,self.r_min)
        return r    
    def function(self, x, y, theta_E, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :return: lensing potential
        """
        x_,y_ = self._dxdy(x,y,center_x,center_y)
        
        r   = self._r(x_, y_,center_x,center_y)
        phi = theta_E[:,np.newaxis]**2*np.log(r)
        phi = np.sum(phi,axis=0)
        return phi

    def derivatives(self, x, y, theta_E, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :return: deflection angle (in angles)
        """
        if len(theta_E)>1e3:
            #return _derivatives_numba(x, y, theta_E, theta_c, center_x, center_y)
            chunk_len = choose_chunk_size(x.size)
            return _derivatives_numba_chunked_lens_parallel(x, y, theta_E, self.r_min, center_x, center_y, chunk_len=chunk_len)

        x_,y_ = self._dxdy(x,y,center_x,center_y)
        r   = self._r(x_, y_,center_x,center_y)        
        alpha   = theta_E[:,np.newaxis]**2/r
        alpha_x = np.sum(alpha*x_/r,axis=0)
        alpha_y = np.sum(alpha*y_/r,axis=0)
        return alpha_x,alpha_y

    def hessian(self, x, y, theta_E, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :return: hessian matrix (in angles)
        """
        if len(theta_E)>1e3:
            #return _derivatives_numba(x, y, theta_E, theta_c, center_x, center_y)
            chunk_len = choose_chunk_size(x.size)
            return _hessian_numba_chunked_lens_parallel(x, y, theta_E, center_x, center_y,chunk_len=chunk_len)
        
        C = theta_E[:,np.newaxis]**2
        x_,y_ = self._dxdy(x, y,center_x,center_y)
        r2    = self._r(x_, y_,center_x,center_y)**2
        # we have to sum over all particles
        f_xx  = np.sum(C * (y_**2-x_**2)/r2**2,axis=0)
        f_yy  = np.sum(C * (x_**2-y_**2)/r2**2,axis=0)
        f_xy  = np.sum(-C * 2*x_*y_/r2**2,axis=0)
        return f_xx, f_xy, f_xy, f_yy
