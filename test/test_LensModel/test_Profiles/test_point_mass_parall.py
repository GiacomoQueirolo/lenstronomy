__author__ = "giacomoqueirolo"
# adapted by test_point_mass.py


from lenstronomy.LensModel.Profiles.point_mass_parall import ParallelPointMass

import numpy as np
import numpy.testing as npt
import pytest


class TestParallelPointMass(object):
    """Tests the ParallelPointMass class routines.

    Note: due to the vectorisations of inputs, we have rounding error
    on the order of machine precision. Test are run with allclose
    instead of assert and == .
    """

    def setup_method(self):
        self.parallpointmass = ParallelPointMass()

    def test_function(self):
        x = np.array([0])
        y = np.array([1])
        theta_E = np.array([1.0])
        values = self.parallpointmass.function(x, y, theta_E)
        npt.assert_allclose(values[0], 0, rtol=1e-12, atol=1e-15)

        x = np.array([0])
        y = np.array([0])
        values = self.parallpointmass.function(x, y, theta_E)
        assert values[0] < 0

        x = np.array([1, 3, 4])
        y = np.array([0, 1, 1])
        values = self.parallpointmass.function(x, y, theta_E)
        npt.assert_allclose(values[0], 0, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[1], 1.151292546497023, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[2], 1.4166066720281081, rtol=1e-12, atol=1e-15)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([0])
        theta_E = np.array([1.0])
        f_x, f_y = self.parallpointmass.derivatives(x, y, theta_E)
        npt.assert_allclose(f_x[0], 1, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(f_y[0], 0, rtol=1e-12, atol=1e-15)
        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.parallpointmass.derivatives(x, y, theta_E)
        npt.assert_allclose(f_x[0], 0, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(f_y[0], 0, rtol=1e-12, atol=1e-15)

        x = np.array([1, 3, 4])
        y = np.array([0, 1, 1])
        values = self.parallpointmass.derivatives(x, y, theta_E)
        npt.assert_allclose(values[0][0], 1, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[1][0], 0, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[0][1], 0.29999999999999999, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[1][1], 0.099999999999999992, rtol=1e-12, atol=1e-15)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([0])
        theta_E = np.array([1.0])
        f_xx, f_xy, f_yx, f_yy = self.parallpointmass.hessian(x, y, theta_E)
        npt.assert_almost_equal(f_xy, f_yx, decimal=8)
        npt.assert_allclose(f_xx[0], -1, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(f_yy[0], 1, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(f_xy[0], -0, rtol=1e-12, atol=1e-15)

        x = np.array([1, 3, 4])
        y = np.array([0, 1, 1])
        values = self.parallpointmass.hessian(x, y, theta_E)
        npt.assert_allclose(values[0][0], -1, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[3][0], 1, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[1][0], -0, rtol=1e-12, atol=1e-15)

        npt.assert_allclose(values[0][1], -0.080000000000000002, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[3][1], 0.080000000000000002, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[1][1], -0.059999999999999998, rtol=1e-12, atol=1e-15)

    def test_mass_3d_lens(self):
        theta_E = np.array([0.5])
        r = 5

        mass_3d = self.parallpointmass.mass_3d_lens(r, theta_E)
        npt.assert_allclose(mass_3d, np.pi * theta_E**2, rtol=1e-12, atol=1e-15)


if __name__ == "__main__":
    pytest.main()
