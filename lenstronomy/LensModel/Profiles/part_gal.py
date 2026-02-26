__author__ = "giacomo_queirolo"

import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase


from particle_galaxy import Gal2MXYZ
from generate_particle_lens import LensPart
from python_tools.conversion import find_index

__all__ = ["PartGal"]


# decorator:
def bounds_error(func):
    """Raise error if given coordinates are outside the bounds of the lenspart area."""

    def func_bounded(self, x, y, *args):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        extents = self.kw_extents["extent_arcsec"]
        if (
            np.any(x < extents[0])
            or np.any(x > extents[1])
            or np.any(y < extents[2])
            or np.any(y > extents[3])
        ):
            raise RuntimeError("Input coordinates are outside allowed range")
        return func(self, x, y, *args)

    return func_bounded


class PartGal(LensProfileBase):
    """Lens profile obtained from particles of a galaxy obtained by simulation."""

    param_names = [""]
    lower_limit_default = {}
    upper_limit_default = {}

    def __init__(self, kwargs_lenspart=None, compute=False):
        """
        kwargs_lenspart: input parameter of the LensPart class
        compute: boolean flag, if True and precomputed results are not available,
            it will compute the lensing properties of the galaxy
        """
        if kwargs_lenspart is None:
            # maybe this error is superflous, just don't define a default value for it...
            raise RuntimeError(
                "The PartGal has to be initialised with the LensPart keywords"
            )
        self.lenspart = LensPart(**kwargs_lenspart)
        if not compute and not self.lenspart.is_precomputed:
            raise RuntimeError(
                "This galaxy was not precomputed. Either run it and store it a-priori, or set compute=True."
            )
        self.lenspart.run()
        # useful for bound_errors
        self.kw_extents = self.lenspart.kw_extents

        super(PartGal, self).__init__()

    @bounds_error
    def get_xy_indexes(self, x, y):
        ra, dec = self.lenspart.get_RADEC()
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))

        index_x = find_index(x.ravel(), ra[0])
        index_y = find_index(y.ravel(), dec[:, 0])
        xy_indexes = np.stack([index_y, index_x], -1).T
        return xy_indexes

    @bounds_error
    def interp_map(self, x, y, map):
        """Interpolate a given map at the given coordinates."""
        xy_indexes = self.get_xy_indexes(x, y)
        int_map = map_coordinates(map, xy_indexes, order=3, mode="nearest")
        return int_map

    @bounds_error
    def function(self, x, y):
        """
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :return: lensing potential
        """
        phi = self.interp_map(self.lenspart.psi, x, y)
        return phi

    @bounds_error
    def derivatives(self, x, y):
        """
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :return: deflection angle (in angles)
        """
        alpha_map = self.alpha_map
        alpha_x = self.lenspart.interp_map(x, y, map_alpha_part_x)
        alpha_y = self.lenspart.interp_map(x, y, map_alpha_part_y)
        return alpha_x, alpha_y

    @bounds_error
    def hessian(self, x, y):
        """
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :return: hessian matrix (in angles)
        """
        f_xx, f_xy, f_yx, f_yy = self.lenspart.hessian
        f_xx = self.lenspart.interp_map(x, y, f_xx)
        f_xy = self.lenspart.interp_map(x, y, f_xy)
        f_yx = self.lenspart.interp_map(x, y, f_yx)
        f_yy = self.lenspart.interp_map(x, y, f_yy)
        return f_xx, f_xy, f_xy, f_yy

    def mass_3d_lens(self, r):
        """Mass enclosed within a 3d sphere of radius r (in angular units).

        :param r: radius in arcsec
        :type r: float
        :return: mass in units of M_sun
        :rtype: float
        """

        m, x, y, z = Gal2MXYZ(self.lenspart.Gal)
        arcXkpc = self.lenspart.arcXkpc
        x_, y_, z_ = x * arcXkpc, y * arcXkpc, z * arcXkpc
        R = np.linalg.norm([x.value, y.value, z.value]) * x.unit
        R_arcsec = R * arcXkpc
        m_ = m[np.where(r < R_arcsec.value)]
        m3d = np.sum(m_)
        m3d_kpc2 = m3d / self.lenspart.SigCrit
        m3d_arc2 = m3d_kpc2 * (arcXkpc**2)
        mass_3d += m3d_arc2.value
        return mass_3d

    def density_lens(self, r):
        """Calculates the 3D density.

        The integral is projected in units of angles (i.e. arc seconds) results in the
        convergence quantity.

        :param r: radius
        :returns: 3D density
        """
        m3d_arc2 = self.mass_3d_lens(r)
        dens = m3d_arc2.value / (np.pi * r * r)
        return dens
