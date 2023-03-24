# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 20:59:58 2022

"""
import matplotlib.pyplot as plt

from reflection_coeffs import ReflectionModel
from reflection_fitting import rs_fit, rs_fit_gap, rp_fit, pygad_fitting
from mlines_data_tools import write_curve_data, read_curve_data,\
    read_curve_metricon, cutoff_data


transfer = read_curve_metricon(file_name='Metricon/Corrected/TE-air.txt',
                               x_lim=(34, 52))
curve = read_curve_metricon(file_name='Metricon/Corrected/T2 TE.txt',
                            x_lim=(34, 52))

# Transfer function
corrected_x = transfer.x
corrected_y = [y1/y2 for y1, y2 in zip(curve.y, transfer.y)]


# Cutting of the data at higher values of y.
# curve = cutoff_data(curve=org_curve, threshold=0.3)

# print(f'Using {int(100*len(curve_x)/len(curve.x))}% of data.')

# params, pcov = curve_fit(rs_fit, curve.x, corrected_y,
#                          bounds=[[420, 20, 1.9, 0], [500, 160, 2, 0.1]])
# curve_fitted = rs_fit(curve.x, params[0], params[1], params[2], params[3])

params, curve_fitted = pygad_fitting(rs_fit, curve.x, corrected_y,
                                      bounds=[[420, 20, 1.9, 0],
                                              [500, 160, 2, 0.1]])

# --------------------------------------
# Setting up the model
model = ReflectionModel(lamb=632.8, n_prism=(2.5822, 2.8639),
                        h_immers=150, h_film=450.0,
                        n_substr=1.515, m_substr=0,
                        n_film=1.9819, m_film=0.00180212556)

# Generating the intensities
int_s = model.Rs_curve(start=transfer.x[0], end=transfer.x[-1], n_points=400)
int_s_ys = model.Rs_curve_fit(curve.x)
# int_p = model.Rp_curve(start=35.5, end=50, n_points=400)
# int_s = model.Rs_curve(start=31, end=45, n_points=400)
# int_p = model.Rp_curve(start=35, end=45, n_points=2000)

# Saving the data
# write_curve_data((curve.x, int_s), 'sokolov_s')
# write_curve_data((curve.x, int_p), 'sokolov_p')

# ------------- Plots -------------
fig, ax = plt.subplots()

ax.plot(int_s.x, int_s.y, label='Model')
# ax.plot(transfer.x, transfer.y, label='Rs (Metricon air)',
#         marker='None')
ax.plot(curve.x, curve.y, label='Rs (Metricon T2)',
        marker='None')

ax.plot(corrected_x, corrected_y, label='Corrected T2 with air')

ax.plot(curve.x, curve_fitted, label='T2 (fitted)')

# ax.plot(int_s.x, int_s.y, label='Rs (Fit)')
ax.set_xlabel("IntAngle")

ax.legend()
ax.grid()
plt.show()
