# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 20:59:58 2022

"""
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from reflection_coeffs import ReflectionModel
from reflection_fitting import rs_fit, rs_fit_gap, rp_fit, pygad_fitting
from mlines_data_tools import write_curve_data, read_curve_data,\
    read_curve_metricon, cutoff_data

# Script parameters
fit_method = 'scipy'
polarization = 's'
background_removal = True

transfer_filename = 'Metricon/Corrected/TE-air.txt'
curve_filename = 'Metricon/Corrected/T2 TE.txt'

# Printing the script parameters. VERBOSE
print( 'Method | Polar. | BG remove')
print(f' {fit_method} |   {polarization}    | {background_removal}\n')

if polarization == 's':
    model_func = rs_fit
elif  polarization == 'p':
    model_func = rp_fit
else:
    raise ValueError(f'The polarization cannot be {polarization}! It can '
                     f'either be \'s\' or \'p\'.')

if polarization == 's':
    transfer = read_curve_metricon(file_name='Metricon/Corrected/TE-air.txt',
                                   x_lim=(34, 52))
    curve = read_curve_metricon(file_name='Metricon/Corrected/T2 TE.txt',
                                x_lim=(34, 52))
    # Printing the curve's file name. VERBOSE
    print(f'Using "{curve_filename}"\n')
    
elif polarization == 'p':
    transfer = read_curve_metricon(file_name='Metricon/Corrected/TM-air.txt',
                                   x_lim=(36.5, 50))
    curve = read_curve_metricon(file_name='Metricon/Corrected/T2 TM.txt',
                                x_lim=(36.5, 50))

# Transfer function
if background_removal:
    corrected_x = transfer.x
    corrected_y = [y1/y2 for y1, y2 in zip(curve.y, transfer.y)]
else:
    corrected_x = curve.x
    corrected_y = curve.y

# Cutting of the data at higher values of y.
# curve = cutoff_data(curve=org_curve, threshold=0.3)

# print(f'Using {int(100*len(curve_x)/len(curve.x))}% of data.')

if fit_method == 'scipy':
    params, pcov = curve_fit(model_func, curve.x, corrected_y,
                              bounds=[[420, 20, 1.9, 0], [500, 160, 2, 0.1]],
                              verbose=2)
    curve_fitted = rs_fit(curve.x, params[0], params[1], params[2], params[3])
elif fit_method == 'pygad':
    params, curve_fitted = pygad_fitting(model_func, curve.x, corrected_y,
                                          bounds=[[420, 20, 1.9, 0],
                                                  [500, 160, 2, 0.1]])
else:
    raise ValueError(f'The fitting method cannot be {fit_method}! It can '
                     f'either be \'scipy\' or \'pygad\'.')

# --------------------------------------
# Setting up the model
model = ReflectionModel(lamb=632.8, n_prism=(2.5822, 2.8639),
                        h_immers=150, h_film=450.0,
                        n_substr=1.515, m_substr=0,
                        n_film=1.9819, m_film=0.00180212556)

# Generating the intensities
if polarization == 's':
    model_curve = model.Rs_curve(start=transfer.x[0], end=transfer.x[-1],
                                 n_points=400)
elif polarization == 'p':
    model_curve = model.Rp_curve(start=transfer.x[0], end=transfer.x[-1],
                                 n_points=400)

# int_p = model.Rp_curve(start=35.5, end=50, n_points=400)
# int_s = model.Rs_curve(start=31, end=45, n_points=400)
# int_p = model.Rp_curve(start=35, end=45, n_points=2000)

# Saving the data
# write_curve_data((curve.x, int_s), 'sokolov_s')
# write_curve_data((curve.x, int_p), 'sokolov_p')

# ------------- Results -------------
print('\n')
print(f'Fitted parameters = {params}')

fig, ax = plt.subplots()

ax.plot(model_curve.x, model_curve.y, label='Model')
# ax.plot(transfer.x, transfer.y, label='Rs (Metricon air)',
#         marker='None')
ax.plot(curve.x, curve.y, label=f'R{polarization} (Metricon T2)',
        marker='None')

if background_removal:
    ax.plot(corrected_x, corrected_y, label='Corrected T2 with air')

ax.plot(curve.x, curve_fitted, label='T2 (fitted)')

# ax.plot(int_s.x, int_s.y, label='Rs (Fit)')
ax.set_xlabel("IntAngle")

ax.legend()
ax.grid()
plt.show()
