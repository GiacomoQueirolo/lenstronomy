__author__ = "giacomoqueirolo"


from lenstronomy.LensModel.Profiles.arsinh import Arsinh

import numpy as np
import numpy.testing as npt
import pytest


class TestArsinh(object):
    """Tests the Arsinh class routines."""

    def setup_method(self):
        self.arsinh = Arsinh()

    def test_function(self):
        x = np.array([0])
        y = np.array([1])
        theta_E = np.array([1.0])
        theta_c = np.array([0.01])

        values = self.arsinh.function(x, y, theta_E, theta_c)
        npt.assert_allclose(values[0], 4.951743777518064, rtol=1e-12, atol=1e-15)

        x = np.array([0])
        y = np.array([0])
        values = self.arsinh.function(x, y, theta_E, theta_c)
        npt.assert_allclose(values[0], 0, rtol=1e-12, atol=1e-15)

        x = np.array([1, 3, 4])
        y = np.array([0, 1, 1])
        values = self.arsinh.function(x, y, theta_E, theta_c)
        npt.assert_allclose(values[0], 4.951743777518064, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[1], 6.103036322777587, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[2], 6.368350448300498, rtol=1e-12, atol=1e-15)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([0])
        theta_E = np.array([1.0])
        theta_c = np.array([0.01])

        f_x, f_y = self.arsinh.derivatives(x, y, theta_E, theta_c)
        npt.assert_allclose(f_x[0], 0.999999995, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(f_y[0], 0.0, rtol=1e-12, atol=1e-15)

        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.arsinh.derivatives(x, y, theta_E, theta_c)
        npt.assert_allclose(f_x[0], 0.0, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(f_y[0], 0.0, rtol=1e-12, atol=1e-15)

        x = np.array([1, 3, 4])
        y = np.array([0, 1, 1])
        values = self.arsinh.derivatives(x, y, theta_E, theta_c)
        npt.assert_allclose(values[0][0], 0.999999995, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[0][1], 0.29999999998499993, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[0][2], 0.23529411764298797, rtol=1e-12, atol=1e-15)

        npt.assert_allclose(values[1][0], 0.0, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[1][1], 0.09999999999499998, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[1][2], 0.05882352941074699, rtol=1e-12, atol=1e-15)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([0])
        theta_E = np.array([1.0])
        theta_c = np.array([0.01])

        f_xx, f_xy, f_yx, f_yy = self.arsinh.hessian(x, y, theta_E, theta_c)
        npt.assert_almost_equal(f_xy, f_yx, decimal=8)
        npt.assert_allclose(f_xx[0], -0.9999999750000005, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(f_xy[0], 0, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(f_yy[0], 0.9999999950000001, rtol=1e-12, atol=1e-15)

        x = np.array([1, 3, 4])
        y = np.array([0, 1, 1])
        values = self.arsinh.hessian(x, y, theta_E, theta_c)
        npt.assert_allclose(values[1], values[2], rtol=1e-12, atol=1e-15)

        npt.assert_allclose(values[0][0], -0.9999999750000005, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[0][1], -0.07999999997799996, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[0][2], -0.05190311418212186, rtol=1e-12, atol=1e-15)

        npt.assert_allclose(values[1][0], 0.0, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[1][1], -0.059999999990999975, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[1][2], -0.027681660898217213, rtol=1e-12, atol=1e-15)

        npt.assert_allclose(values[3][0], 0.9999999950000001, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[3][1], 0.07999999999799996, rtol=1e-12, atol=1e-15)
        npt.assert_allclose(values[3][2], 0.051903114186192686, rtol=1e-12, atol=1e-15)


if __name__ == "__main__":
    pytest.main()
