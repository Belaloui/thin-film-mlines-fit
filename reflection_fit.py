# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 20:59:58 2022

"""
from collections import namedtuple
import matplotlib.pyplot as plt
import datetime
import shutil
import os

import argparse

import numpy as np
from scipy.optimize import curve_fit

from reflection_fitting import pygad_fitting, ModelFunction
from mlines_data_tools import write_curve_data,\
    read_curve_metricon, read_config
from reflection_stats import r_squared

# ---------- Parser ----------
parser = argparse.ArgumentParser()
parser.add_argument('--config', metavar='Configuration file', type=str,
                    required=True,
                    help="The path to the model configuration file")
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

config_filename = args.config
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

if polarization not in ['s', 'p']:
    raise ValueError(f'The polarization cannot be {polarization}! It can '
                     f'either be \'s\' or \'p\'.')

# Loading model's configuration
variables = ['h_immers', 'h_film', 'n_substr', 'm_substr',
             'n_film', 'm_film']
fixed, bounds_dict, x_limits = read_config(config_filename)

for f in fixed:
    variables.pop(variables.index(f))

bounds = [0]*len(bounds_dict)
for key in bounds_dict:
    bounds[variables.index(key)] = bounds_dict[key]

min_bounds = []
max_bounds = []

for min_b, max_b in bounds:
    min_bounds.append(min_b)
    max_bounds.append(max_b)

curve = read_curve_metricon(file_name=curve_filename,
                            x_lim=x_limits)

# Applying the background removal if a transfer curve is provided.
if background_removal:
    transfer = read_curve_metricon(file_name=transfer_filename,
                                   x_lim=x_limits)
    corrected_x = transfer.x
    corrected_y = [y1/y2 for y1, y2 in zip(curve.y, transfer.y)]
else:
    corrected_x = curve.x
    corrected_y = curve.y

# Creating the model's function
model = ModelFunction(polarization='s',
                      fixed_params=fixed)
model_func = model.model_func

# Fitting ...
if fit_method == 'scipy':
    p0 = [(a+b)/2 for a,b in zip(min_bounds, max_bounds)]
    
    params, pcov = curve_fit(model_func, curve.x, corrected_y,
                              p0=p0,
                              bounds=[min_bounds, max_bounds],
                              verbose=2)
    curve_fitted = model_func(curve.x, *params)
elif fit_method == 'pygad':
    params, curve_fitted = pygad_fitting(model_func, curve.x, corrected_y,
                                          bounds=[min_bounds, max_bounds])
else:
    raise ValueError(f'The fitting method cannot be {fit_method}! It can '
                     f'either be \'scipy\' or \'pygad\'.')

# Computing R^2
r_sqr = r_squared(curve_fitted, corrected_y)

# ------------- Showing the results -------------
print('\n')
print(f'Fitted parameters : {list(bounds_dict.keys())} = {params}\n')
print(f'R^2 = {r_sqr}\n')

fig, ax = plt.subplots()

if background_removal:
    ax.plot(transfer.x, transfer.y, label='Background')
# ax.plot(transfer.x, transfer.y, label='Rs (Metricon air)',
#         marker='None')
ax.plot(curve.x, curve.y, label=f'R{polarization} (Metricon data)',
        marker='None')

if background_removal:
    ax.plot(corrected_x, corrected_y, label='Corrected data')

ax.plot(curve.x, curve_fitted, label='Fitted curve')

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

# Saving the config file
shutil.copyfile(config_filename, f'{res_path}/{time_str}_config.txt')

