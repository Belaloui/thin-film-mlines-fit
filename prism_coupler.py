# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:35:09 2022
"""

import numpy as np
from numpy import cos, sin, sqrt, exp, pi
import matplotlib.pyplot as plt


thetas_deg = np.linspace(43, 60, 4000)
thetas = np.radians(thetas_deg)

lembda = 632.8  # nm
k = 2.*pi/lembda

N_p = 2.14044

H_im = 150  # nm
H_f = 1082  # nm

n_s = 1.45705
m_s = 0.000

n_f = 1.840
m_f = 0.001


eps_f = (n_f + 1j*m_f)**2
eps_s = (n_s + 1j*m_s)**2
eps_im = 1.


def kz(theta):
    return k*N_p*cos(theta)


def ky(theta):
    return k*N_p*sin(theta)


def gam_II(theta):
    return sqrt(ky(theta)**2 - k**2*eps_im + 0j)


def gam_III(theta):
    return sqrt(ky(theta)**2 - k**2*eps_f + 0j)


def gam_IV(theta):
    return sqrt(ky(theta)**2 - k**2*eps_s + 0j)


def As(theta):
    II_III = (gam_II(theta)-gam_III(theta))/(gam_II(theta)+gam_III(theta))
    III_IV = (gam_III(theta)-gam_IV(theta))/(gam_III(theta)+gam_IV(theta))

    num = II_III + III_IV * exp(-2*gam_III(theta)*H_f)
    den = 1 + II_III * III_IV * exp(-2*gam_III(theta)*H_f)

    return num/den


def Rs_E(theta):
    kz_II = (kz(theta)-1j*gam_II(theta))/(kz(theta)+1j*gam_II(theta))

    num = kz_II + As(theta)*exp(-2*gam_II(theta)*H_im)
    den = 1 + kz_II * As(theta)*exp(-2*gam_II(theta)*H_im)

    return num/den


def Rs(theta):
    return abs(Rs_E(theta))**2


plt.plot(thetas_deg, [Rs(theta) for theta in thetas], label='Rs')
plt.legend()

