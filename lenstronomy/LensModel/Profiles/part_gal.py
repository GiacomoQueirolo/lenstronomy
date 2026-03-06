__author__ = "giacomo_queirolo"

import numpy as np
from scipy.ndimage import map_coordinates
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase


from python_tools.conversion import find_index

try:
    from nazgul.particle_galaxy import Gal2MXYZ
    from nazgul.mount_doom.generate_particle_lens_sub import SubLensPart
except ModuleNotFoundError as e:
    ModuleNotFoundError(
        str(e)
        + "\nPlease first install nazgul (https://github.com/GiacomoQueirolo/nazgul)"
    )

__all__ = ["Part_Gal"]


def _bound_mask(extents, x, y):
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    mask = np.where(
        (x < extents[0]) | (x > extents[1]) | (y < extents[2]) | (y > extents[3])
    )
    return mask


def check_bounds(extents, x, y):
    mask = _bound_mask(extents, x, y)
    if len(mask[0]) != 0:
        return False
    return True


# decorator:
def bounds_error(func):
    """Raise error if given coordinates are outside the bounds of the lenspart area."""

    def func_bounded(self, x, y, *args):
        extents = self.kw_extents["extent_arcsec"]
        if check_bounds(extents, x, y):
            raise RuntimeError("Input coordinates are outside allowed range")
        return func(self, x, y, *args)

    return func_bounded


class Part_Gal(LensProfileBase):
    """Lens profile obtained from particles of a galaxy obtained by simulation."""

    param_names = ["kwargs_lenspart", "compute", "z_lens", "z_source", "lenspart"]
    lower_limit_default = {}
    upper_limit_default = {}

    def __init__(
        self,
        kwargs_lenspart=None,
        compute=False,
        z_lens=None,
        z_source=None,
        lenspart=None,
    ):
        """
        kwargs_lenspart: input parameter of the LensPart class
        compute: boolean flag, if True and precomputed results are not available,
            it will compute the lensing properties of the galaxy
        """
        if kwargs_lenspart is None and lenspart is None:
            # maybe this error is superflous, just don't define a default value for it...
            raise RuntimeError(
                "The Part_Gal has to be initialised with the LensPart keywords or has to be given an instance of SubLensPart"
            )
        if lenspart is None:
            if not compute:
                kwargs_lenspart["reload"] = True
            self.lenspart = SubLensPart(**kwargs_lenspart)
            if not compute and not self.lenspart.is_precomputed():
                raise RuntimeError(
                    "This galaxy was not precomputed. Either run it and store it 'a priori', or set compute=True."
                )
        else:
            self.lenspart = lenspart
        self.lenspart.run()

        if z_lens is None:
            z_lens = self.lenspart.z_lens
            print(f"Considering z_lens = z_galaxy = {z_lens}")

        if z_source is None:
            # or we could set it to a very high value
            z_source = self.lenspart.z_source
            print(f"Considering z_source = z_source(sampled) = {z_source}")
        else:
            # Verify it is within the bounds
            assert (
                self.lenspart.z_source_min < z_source
                and self.lenspart.z_source_max >= z_source
            )
        self.z_lens = z_lens
        self.z_source = z_source
        # useful for bound_errors
        self.kw_extents = self.lenspart.kw_extents
        super(Part_Gal, self).__init__()

    def get_xy_indexes(self, x, y):
        ra, dec = self.lenspart.get_RADEC()
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))

        index_x = find_index(x.ravel(), ra[0])
        index_y = find_index(y.ravel(), dec[:, 0])
        xy_indexes = np.stack([index_y, index_x], -1).T
        return xy_indexes

    def _interp_map(self, x, y, map):
        """Interpolate a given map at the given coordinates.

        (doesn't check bounds)
        """
        xy_indexes = self.get_xy_indexes(x, y)
        int_map = map_coordinates(map, xy_indexes, order=3, mode="nearest")
        return int_map

    def interp_map_rescale_zlzs(self, x, y, map, map_func):
        """Interpolate a given map at the given coordinates and rescale it for the given
        redshifts."""
        if (
            self.z_lens == self.lenspart.z_lens
            and self.z_source == self.lenspart.z_source
        ):
            return self.interp_map_bounds(x, y, map, map_func)
        Ds_prime = self.lenspart.cosmo.angular_diameter_distance(self.z_source)
        Dd_prime = self.lenspart.cosmo.angular_diameter_distance(self.z_lens)
        Dds_prime = self.lenspart.cosmo.angular_diameter_distance_z1z2(
            self.z_lens, self.z_source
        )

        Ds = self.lenspart.cosmo.angular_diameter_distance(self.lenspart.z_source)
        Dd = self.lenspart.cosmo.angular_diameter_distance(self.lenspart.z_lens)
        Dds = self.lenspart.cosmo.angular_diameter_distance_z1z2(
            self.lenspart.z_lens, self.lenspart.z_source
        )

        x_scaled, y_scaled = x * Dd_prime / Dd, y * Dd_prime / Dd
        # I believe the bounds should be checked on the unscaled coordinates
        int_map = self.interp_map_bounds(x_scaled, y_scaled, map, map_func)
        scale_map = (Dds_prime / Ds_prime) / (Dds / Ds)
        int_map_scaled = scale_map * int_map
        return int_map_scaled

    def interp_map_bounds(self, x, y, map, map_func):
        extents = self.kw_extents["extent_arcsec"]
        # intepolate everything
        map_interpolated = self._interp_map(x, y, map)
        if not check_bounds(extents, x, y):
            # overwrite for the points outside of bounds with exact fit
            mask_OoB = _bound_mask(extents, x, y)
            if len(mask_OoB[0]) > int(1e3):
                raise RuntimeError(
                    f"Too many pixels are outside of bounds: N={len(mask_OoB[0])}"
                )
            map_interpolated[mask_OoB] = map_func(x[mask_OoB], y[mask_OoB])
        return map_interpolated

    def function(self, x, y):
        """
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :return: lensing potential
        """
        psi_map = self.lenspart.psi

        def psi_func(x, y):
            return self.lenspart.lens_prof.function(x, y, **self.lenspart.kwargs_lens)

        psi = self.interp_map_rescale_zlzs(x, y, psi_map, map_func=psi_func)
        return psi

    def derivatives(self, x, y):
        """
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :return: deflection angle (in angles)
        """
        map_alpha_x, map_alpha_y = self.lenspart.alpha_map

        def alpha_func_x(x, y):
            return self.lenspart.lens_prof.derivatives(
                x, y, **self.lenspart.kwargs_lens
            )[0]

        def alpha_func_y(x, y):
            return self.lenspart.lens_prof.derivatives(
                x, y, **self.lenspart.kwargs_lens
            )[1]

        alpha_x = self.interp_map_rescale_zlzs(x, y, map_alpha_x, map_func=alpha_func_x)
        alpha_y = self.interp_map_rescale_zlzs(x, y, map_alpha_y, map_func=alpha_func_y)
        return alpha_x, alpha_y

    def hessian(self, x, y):
        """
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :return: hessian matrix (in angles)
        """
        f_xx, f_xy, f_yx, f_yy = self.lenspart.hessian

        def hessian_func(x, y):
            return self.lenspart.lens_prof.hessian(x, y, **self.lenspart.kwargs_lens)

        def hessian_func_xx(x, y):
            return hessian_func(x, y)[0]

        def hessian_func_xy(x, y):
            return hessian_func(x, y)[1]

        def hessian_func_yx(x, y):
            return hessian_func(x, y)[2]

        def hessian_func_yy(x, y):
            return hessian_func(x, y)[3]

        f_xx = self.interp_map_rescale_zlzs(x, y, f_xx, map_func=hessian_func_xx)
        f_xy = self.interp_map_rescale_zlzs(x, y, f_xy, map_func=hessian_func_xy)
        f_yx = self.interp_map_rescale_zlzs(x, y, f_yx, map_func=hessian_func_yx)
        f_yy = self.interp_map_rescale_zlzs(x, y, f_yy, map_func=hessian_func_yy)
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
