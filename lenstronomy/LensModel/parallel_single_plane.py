import numpy as np
from numba import njit, prange

#from concurrent.futures import as_completed
#from concurrent.futures import ThreadPoolExecutor as Executor

from lenstronomy.LensModel.single_plane import SinglePlane


def _compute_derivatives(args):
    func, x, y, kw = args
    return func.derivatives(x, y, **kw)


@njit(parallel=True, fastmath=True)
def alpha_sum(x, y, theta_E, theta_c, center_x, center_y):
    f_x = np.zeros_like(x)
    f_y = np.zeros_like(y)
    for i in prange(len(theta_E)):
        x_ = x - center_x[i]
        y_ = y - center_y[i]
        r2 = x_**2 + y_**2
        f = theta_E[i]**2 / np.sqrt(theta_c[i]**4 + r2**2)
        f_x += f * x_
        f_y += f * y_
    return f_x, f_y


class ParallelSinglePlane(SinglePlane):
    """Drop-in replacement for lenstronomy.LensModel.single_plane.SinglePlane with parallel alpha()."""

    def alpha(self, x, y, kwargs, k=None, max_workers=None):
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        # Single lens call remains the same
        if isinstance(k, int):
            return self.func_list[k].derivatives(x, y, **kwargs[k])

        bool_list = self._bool_list(k)
        f_x = np.zeros_like(x)
        f_y = np.zeros_like(x)

        # Build task list only for active lenses
        tasks = [(func, x, y, kw)
                 for func, kw, use in zip(self.func_list, kwargs, bool_list)
                 if use]

        if len(tasks) == 1:
            # shortcut for single lens
            fx_i, fy_i = tasks[0][0].derivatives(x, y, **tasks[0][3])
            return fx_i, fy_i
        # special case if many profiles are particle-like
        #if self._model_list

        # Parallel execution
        with Executor(max_workers=max_workers) as executor:
            futures = [executor.submit(_compute_derivatives, t) for t in tasks]
            for f in as_completed(futures):
                fx_i, fy_i = f.result()
                f_x += fx_i
                f_y += fy_i
        return f_x, f_y


    def ray_shooting(self, x, y, kwargs, k=None):
        print("ray shooting pewpew")
        """
        maps image to source position (inverse deflection)
        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: source plane positions corresponding to (x, y) in the image plane
        """

        dx, dy = self.alpha(x, y, kwargs, k=k)

        return x - dx, y - dy
        
        

