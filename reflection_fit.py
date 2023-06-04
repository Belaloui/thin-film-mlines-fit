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
from reflection_stats import r_squared, std_dev

# Hyper params
n_pygad_fits = 5
n_gens = 30

# ---------- Parser ----------
parser = argparse.ArgumentParser()
parser.add_argument('--config', metavar='Configuration file', type=str,
                    required=True,
                    help="The path to the model configuration file.")
parser.add_argument('--curve', metavar='Measurement curve', type=str,
                    required=True,
                    help="The path to the curve's data")
parser.add_argument('--fit_method', metavar='Fitting method', type=str,
                    default='scipy',
                    help=f"Can either be 'scipy', 'pygad', or 'pygad:N_GENS', "
                         f"where N_GENS is the maximum number of generations"
                         f" for the genetic algorithm. By default, it is {n_gens}.")
parser.add_argument('--pol', metavar='Polarization', type=str,
                    required=True,
                    help="Can either be 's' or 'p'.")
parser.add_argument('--rb', metavar='Background curve', type=str,
                    help="When a background curve is provided, it is "
                         "used to remove the measurement curve's backgound.")
parser.add_argument('--std', metavar='Standard deviation', action=argparse.BooleanOptionalAction,
                    help=f"When set, {n_pygad_fits} fittings are performed "
                         f"to get the standard deviations for the GA results.")

args = parser.parse_args()
# ---------- Parser ----------

print('\n    -----  M-Lines fitting script  -----\n')  # Header

# --------------- Script's parameters --------------- #
fit_method_input = args.fit_method
if fit_method_input in ['scipy', 'pygad']:
    fit_method = args.fit_method
elif ':' in fit_method_input:
    fit_method, n_gens = fit_method_input.split(':')
    if fit_method != 'pygad':
        raise ValueError(f'The fitting method cannot be \'{fit_method_input}\'! It can '
                     f'either be \'scipy\' or \'pygad\'.')
    if n_gens.isdigit():
        n_gens = int(n_gens)
    else:
        raise ValueError(f'\'{n_gens}\' is not a valid number of generations!')
else:
    raise ValueError(f'The fitting method cannot be \'{fit_method_input}\'! It can '
                     f'either be \'scipy\' or \'pygad\'.')
polarization = args.pol
background_removal = args.rb is not None

config_filename = args.config
transfer_filename = args.rb
curve_filename = args.curve

# Compute STDs for pygad?
pygad_std = args.std is not None
# --------------- Script's parameters --------------- #

# Printing the curve's file name. VERBOSE
print(f'Using "{curve_filename}"\n')
# Printing the curve's file name. VERBOSE
if background_removal:
    print(f'With background "{transfer_filename}"\n')

# Printing the script parameters. VERBOSE
print( 'Method | Polar. | BG remove')
print(f' {fit_method} |   {polarization}    | {background_removal}')
if(fit_method == 'pygad'):
    print(f'{n_gens} generations per fitting.\n')
print('\n')

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
    if not pygad_std:
        params, curve_fitted = pygad_fitting(model_func, curve.x, corrected_y,
                                             bounds=[min_bounds, max_bounds],
                                             n_gens=n_gens)
    else:
        pygad_res = [pygad_fitting(model_func, curve.x, corrected_y,
                                   bounds=[min_bounds, max_bounds],
                                             n_gens=n_gens)
                        for _ in range(n_pygad_fits)]
        params_list = [elem[0] for elem in pygad_res]
        curve_list = [elem[1] for elem in pygad_res]
        
        params = np.mean(params_list, axis=0)
        curve_fitted = np.mean(curve_list, axis=0)

# Computing R^2
r_sqr = r_squared(curve_fitted, corrected_y)

# Computing standard deviations and 95% CIs
perr = None
if fit_method == 'scipy':
    perr = std_dev(pcov)
    ci_95 = [(p-1.96*e, p+1.96*e) for p, e in zip(params, perr)]
elif fit_method == 'pygad' and pygad_std:
    perr = np.std(params_list, axis=0)
    ci_95 = [(p-1.96*e, p+1.96*e) for p, e in zip(params, perr)]


# ------------- Showing the results -------------
print('\n')
print(f'Fitted parameters : {list(bounds_dict.keys())} = {params}\n')
if perr is not None:
    print(f'std_devs : {perr}')
    print(f'95% CI : {ci_95}')

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
time_str = dt.strftime("%Y-%m-%d-%H-%M-%S")

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

# Saving errors and intervals
if perr is not None:
    np.savetxt(f'{res_path}/{time_str}_std_devs.txt', perr,
               delimiter=', ')
    np.savetxt(f'{res_path}/{time_str}_95_CI.txt', ci_95,
               delimiter=', ')


# Saving all outputs
with open(f'{res_path}/{time_str}_output.txt', 'w') as out_file:
    # Printing the curve's file name. VERBOSE
    out_file.write(f'Using "{curve_filename}"\n')
    # Printing the curve's file name. VERBOSE
    if background_removal:
        out_file.write(f'With background "{transfer_filename}"\n')

    # Printing the script parameters. VERBOSE
    out_file.write( 'Method | Polar. | BG remove\n')
    out_file.write(f' {fit_method} |   {polarization}    | {background_removal}\n')
    
    out_file.write('\n')
    out_file.write(f'Fitted parameters : {list(bounds_dict.keys())} = {params}\n')
    if perr is not None:
        out_file.write(f'std_devs : {perr}\n')
        out_file.write(f'95% CI : {ci_95}\n')

    out_file.write(f'R^2 = {r_sqr}\n') 

# Saving the config file
shutil.copyfile(config_filename, f'{res_path}/{time_str}_config.txt')

