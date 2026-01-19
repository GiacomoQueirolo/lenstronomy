__author__ = "pierrefleury"

import numpy as np

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
#from scipy.special import hyp2f1

__all__ = ["Arsinh"]


class Arsinh(LensProfileBase):
    """
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
        super(Arsinh, self).__init__()


    def function(self, x, y, theta_E, theta_c, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :param theta_c: core radius (in angles)
        :return: lensing potential (in squared angles)
        """
        x_ = x - center_x
        y_ = y - center_y
        r2 = x_**2 + y_**2
        
        psi = 0.5 * theta_E**2 * np.arcsinh(r2 / theta_c**2)
        
        return psi


    def derivatives(self, x, y, theta_E, theta_c, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :param theta_c: core radius (in angles)
        :return: displacement angle (in angles)
        """
        x_ = x - center_x
        y_ = y - center_y
        r2 = x_**2 + y_**2
        
        f = theta_E**2 / np.sqrt(theta_c**4 + r2**2)
        
        alpha_x = f * x_
        alpha_y = f * y_
        
        return alpha_x, alpha_y

    def hessian(self, x, y, theta_E, theta_c, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :param theta_c: core radius (in angles)
        :return: Hessian matrix (dimensionless)
        """
        x_ = x - center_x
        y_ = y - center_y
        r2 = x_**2 + y_**2
        
        l4 = theta_E**2 / (theta_c**4 + r2**2)**(1.5)
        kappa = l4 * theta_c**4
        gamma1 = - l4 * r2 * (x_**2 - y_**2)
        gamma2 = - l4 * r2 * 2 * x_ * y_
        
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        
        return f_xx, f_xy, f_xy, f_yy
