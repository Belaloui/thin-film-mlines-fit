# (C) Copyright Nacer eddine Belaloui, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from collections import namedtuple
from collections.abc import Iterable
import numpy as np
from numpy import cos, sin, sqrt, exp, pi


class ReflectionModel:
    """ The model for the reflection coefficient in a prism coupler as
        described in Sokolov et al. (eqts. 5, 10)
        It computes the Rs, Rp coeffs either within a range for angles
        (in degrees) or for a predefined set of angles (in degrees).

        Units: All lengths are in nanometers.
    """

    def __init__(self, lamb, n_prism, h_immers, h_film,
                 n_substr, m_substr, n_film, m_film):
        """ n_prism = np or (np_o, np_e).
        """
        # Light
        self.lembda = lamb  # nm
        self.k = 2.*pi/self.lembda

        # Prism
        self.N_p = n_prism
        if isinstance(self.N_p, Iterable):  # Handeling anisotropic prisms
            self.N_po = self.N_p[0]  # Ordinary refraction
            self.N_pe = self.N_p[1]  # Extaordinary refraction
        else:
            self.N_po = self.N_p  # For isotropic prisms, it's the same Np
            self.N_pe = self.N_p

        # Thicknesses
        self.H_im = h_immers  # nm
        self.H_f = h_film  # nm

        # Substrate
        self.n_s = n_substr
        self.m_s = m_substr

        # Film
        self.n_f = n_film
        self.m_f = m_film

        # Permittivities
        self.eps_f = (self.n_f + 1j*self.m_f)**2
        self.eps_s = (self.n_s + 1j*self.m_s)**2
        self.eps_im = 1.
        self.eps_po = self.N_po**2
        self.eps_pe = self.N_pe**2

    def kz(self, theta, polarization):
        if polarization == 's':
            return self.k*self.N_pe*cos(theta)
        if polarization == 'p':
            return self.k*self.N_po*cos(theta)

    def ky(self, theta, polarization):
        if polarization == 's':
            return self.k*self.N_pe*sin(theta)
        if polarization == 'p':
            return self.k*self.N_po*sin(theta)

    def gam_II(self, theta, polarization):
        return sqrt(self.ky(theta, polarization)**2
                    - self.k**2*self.eps_im + 0j)

    def gam_III(self, theta, polarization):
        return sqrt(self.ky(theta, polarization)**2
                    - self.k**2*self.eps_f + 0j)

    def gam_IV(self, theta, polarization):
        return sqrt(self.ky(theta, polarization)**2
                    - self.k**2*self.eps_s + 0j)

    # As and Rs/E are described in eqt. 5 in Sokolov et al.
    def As(self, theta):
        II_III = ((self.gam_II(theta, 's')-self.gam_III(theta, 's')) /
                  (self.gam_II(theta, 's')+self.gam_III(theta, 's')))
        III_IV = ((self.gam_III(theta, 's')-self.gam_IV(theta, 's')) /
                  (self.gam_III(theta, 's')+self.gam_IV(theta, 's')))

        num = II_III + III_IV * exp(-2*self.gam_III(theta, 's')*self.H_f)
        den = 1 + II_III * III_IV * exp(-2*self.gam_III(theta, 's')*self.H_f)

        return num/den

    def Rs_E(self, theta):
        kz_II = ((self.kz(theta, 's')-1j*self.gam_II(theta, 's')) /
                 (self.kz(theta, 's')+1j*self.gam_II(theta, 's')))

        num = kz_II + self.As(theta) \
            * exp(-2*self.gam_II(theta, 's')*self.H_im)
        den = 1 + kz_II * self.As(theta) \
            * exp(-2*self.gam_II(theta, 's')*self.H_im)

        return num/den

    def Rs(self, theta):
        """ Computes the Rs coefficient for a single angle (in radians).
        """
        return abs(self.Rs_E(theta))**2

    def Rs_curve(self, start, end, n_points=4000):
        """ Computes the Rs coefficient for a range of angles using
        n_points angle.
        """
        thetas_deg = np.linspace(start, end, n_points)
        thetas = np.radians(thetas_deg)
        Curve = namedtuple('Curve', 'x y')
        return Curve(thetas_deg, [self.Rs(theta) for theta in thetas])

    def Rs_curve_fit(self, angles):
        """ Computes the Rs coefficient for a predefined set of angles (in
            degrees).
            Useful for curve fitting.
        """
        thetas = np.radians(angles)
        return [self.Rs(theta) for theta in thetas]

    # Ap and Rp/H are described in eqt. 10 in Sokolov et al.
    def Ap(self, theta):
        II_III = ((self.gam_II(theta, 'p')*self.eps_f-self.gam_III(theta, 'p')*self.eps_im) /
                  (self.gam_II(theta, 'p')*self.eps_f+self.gam_III(theta, 'p')*self.eps_im))
        III_IV = ((self.gam_III(theta, 'p')*self.eps_s-self.gam_IV(theta, 'p')*self.eps_po) /
                  (self.gam_III(theta, 'p')*self.eps_s+self.gam_IV(theta, 'p')*self.eps_po))

        num = II_III + III_IV * exp(-2*self.gam_III(theta, 'p')*self.H_f)
        den = 1 + II_III * III_IV * exp(-2*self.gam_III(theta, 'p')*self.H_f)

        return num/den

    def Rp_H(self, theta):
        kz_II = ((self.kz(theta, 'p')*self.eps_im-1j*self.gam_II(theta, 'p')*self.eps_po) /
                 (self.kz(theta, 'p')*self.eps_im+1j*self.gam_II(theta, 'p')*self.eps_po))

        num = kz_II + self.Ap(theta)*exp(-2*self.gam_II(theta, 'p')*self.H_im)
        den = 1 + kz_II * self.Ap(theta)*exp(-2*self.gam_II(theta, 'p')*self.H_im)

        return num/den

    def Rp(self, theta):
        """ Computes the Rp coefficient for a single angle (in radians).
        """
        return abs(self.Rp_H(theta))**2

    def Rp_curve(self, start, end, n_points=4000):
        """ Computes the Rp coefficient for a range of angles using
        n_points angle.
        """
        thetas_deg = np.linspace(start, end, n_points)
        thetas = np.radians(thetas_deg)
        Curve = namedtuple('Curve', 'x y')
        return Curve(thetas_deg, [self.Rp(theta) for theta in thetas])

    def Rp_curve_fit(self, angles):
        """ Computes the Rp coefficient for a predefined set of angles (in
            degrees).
            Useful for curve fitting.
        """
        thetas = np.radians(angles)
        return [self.Rp(theta) for theta in thetas]

    def Ts_ap(self, angles):
        thetas = np.radians(angles)
        return sqrt(self.N_pe**2 - 1*sin(thetas)**2)/(1*cos(thetas))

    def Ts_pa(self, angles):
        thetas = np.radians(angles)
        return (1*cos(thetas))/sqrt(self.N_pe**2 - 1*sin(thetas)**2)

    def Tp_ap(self, angles):
        thetas = np.radians(angles)
        return sqrt(self.N_po**2 - 1*sin(thetas)**2)/(1*cos(thetas))

    def Tp_pa(self, angles):
        thetas = np.radians(angles)
        return (1*cos(thetas))/sqrt(self.N_po**2 - 1*sin(thetas)**2)
