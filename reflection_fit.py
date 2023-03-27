# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 20:59:58 2022

"""
from collections import namedtuple
import matplotlib.pyplot as plt
import datetime
import os

import argparse

import numpy as np
from scipy.optimize import curve_fit

from reflection_coeffs import ReflectionModel
from reflection_fitting import rs_fit, rs_fit_gap, rp_fit, pygad_fitting,\
    ModelFunction
from mlines_data_tools import write_curve_data, read_curve_data,\
    read_curve_metricon, cutoff_data

# ---------- Parser ----------
parser = argparse.ArgumentParser()
parser.add_argument('--curve', metavar='Measurement curve', type=str,
                    required=True,
                    help="The path to the curve's data")
parser.add_argument('--fit_method', metavar='Fitting method', type=str,
                    default='scipy',
                    help="Can either be 'scipy' or 'pygad'")
parser.add_argument('--pol', metavar='Polarization', type=str,
                    required=True,
                    help="Can either be 's' or 'p'")
parser.add_argument('--rb', metavar='Background curve', type=str,
                    help="When a background curve is provided, it is "
                         "used to remove the measurement curve's backgound")
args = parser.parse_args()
# ---------- Parser ----------

print('\n    -----  M-Lines fitting script  -----\n')  # Header

# Script's parameters
fit_method = args.fit_method
polarization = args.pol
background_removal = args.rb is not None

transfer_filename = args.rb
curve_filename = args.curve

# Printing the curve's file name. VERBOSE
print(f'Using "{curve_filename}"\n')
# Printing the curve's file name. VERBOSE
if background_removal:
    print(f'With background "{transfer_filename}"\n')

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

curve = read_curve_metricon(file_name=curve_filename,
                            x_lim=(32, 52))

# Applying the background removal if a transfer curve is provided.
if background_removal:
    transfer = read_curve_metricon(file_name=transfer_filename,
                                   x_lim=(32, 52))
    corrected_x = transfer.x
    corrected_y = [y1/y2 for y1, y2 in zip(curve.y, transfer.y)]
else:
    corrected_x = curve.x
    corrected_y = curve.y
    
# Creating the model's function
model = ModelFunction(polarization='s',
                      fixed_params={'n_substr':1.51, 'm_substr':0})
model_fun = model.model_func


# Fitting ...
if fit_method == 'scipy':
    params, pcov = curve_fit(model_func, curve.x, corrected_y,
                              bounds=[[200, 20, 1.9, 0], [300, 160, 2, 0.1]],
                              verbose=2)
    curve_fitted = rs_fit(curve.x, params[0], params[1], params[2], params[3])
elif fit_method == 'pygad':
    params, curve_fitted = pygad_fitting(model_func, curve.x, corrected_y,
                                          bounds=[[420, 20, 1.9, 0],
                                                  [500, 160, 2, 0.1]])
else:
    raise ValueError(f'The fitting method cannot be {fit_method}! It can '
                     f'either be \'scipy\' or \'pygad\'.')


# ------------- Showing the results -------------
print('\n')
print(f'Fitted parameters = {params}')

fig, ax = plt.subplots()

ax.plot(transfer.x, transfer.y, label='Background')
# ax.plot(transfer.x, transfer.y, label='Rs (Metricon air)',
#         marker='None')
ax.plot(curve.x, curve.y, label=f'R{polarization} (Metricon T2)',
        marker='None')

if background_removal:
    ax.plot(corrected_x, corrected_y, label='Corrected T2 with air')

ax.plot(curve.x, curve_fitted, label='T2 (fitted)')

# ax.plot(int_s.x, int_s.y, label='Rs (Fit)')
ax.set_xlabel("Internal Angle")
ax.set_ylabel("Normalized Intensity")

ax.legend()
ax.grid()
plt.show()

# ---------- Saving Results ----------- #

# Get date and time for files names
dt = datetime.datetime.now()
time_str = dt.strftime("%Y%m%d%H%M%S")

# Creating the results folder
res_path = f'{time_str}_results'
if not os.path.exists(res_path):
   os.makedirs(res_path)

# Save the plot as an EPS file
fig.savefig(f'{res_path}/{time_str}_curves.eps', format='eps')

# Save the fitted curve
Curve = namedtuple('Curve', 'x y')
result_curve = Curve(curve.x, curve_fitted)
write_curve_data(result_curve, f'{res_path}/{time_str}_fitted_curve')

# Saving the fitted parameters and covariances
np.savetxt(f'{res_path}/{time_str}_fitted_parameters.txt', params,
           delimiter=', ')
if fit_method == 'scipy':
    np.savetxt(f'{res_path}/{time_str}_covariances.txt', pcov,
               delimiter=', ')
