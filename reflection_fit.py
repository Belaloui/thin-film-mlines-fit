# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 20:59:58 2022

"""
from reflection_coeffs import ReflectionModel
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pygad as pg
from mlines_data_tools import write_curve_data, read_curve_data,\
    read_curve_metricon, cutoff_data


# ------------- Fitting tools ------------
def rs_fit(angles, hf, him, nf, mf):
    """ The fitted Rs function. Used to fit Hf, nf, and mf given a
    list of angles (in degrees).
    """
    model = ReflectionModel(lamb=632.8, n_prism=(2.5822, 2.8639),
                            h_immers=him, h_film=hf,
                            n_substr=1.51, m_substr=0,
                            n_film=nf, m_film=mf)
    rs = model.Rs_curve_fit(angles)
    return rs


def rp_fit(angles, hf, him, nf, mf):
    """ The fitted Rp function. Used to fit Hf, nf, and mf given a
    list of angles (in degrees).
    """
    model = ReflectionModel(lamb=632.8, n_prism=(2.5822, 2.8639),
                            h_immers=him, h_film=hf,
                            n_substr=1.515, m_substr=0,
                            n_film=nf, m_film=mf)
    rp = model.Rp_curve_fit(angles)
    return rp


def rs_fit_gap(angles, him):
    """ The fitted Rs function for the gap. Used to fit H_im given a
    list of angles (in degrees).
    """
    model = ReflectionModel(lamb=632.8, n_prism=(2.5822, 2.8639),
                            h_immers=him, h_film=450,
                            n_substr=1.515, m_substr=0,
                            n_film=1.9819, m_film=0.002)
    rs = model.Rs_curve_fit(angles)
    return rs


def pygad_fitting(model_func, x_data, y_data, bounds):

    def fitness(solution, solution_idx):
        output = model_func(angles=x_data,
                            hf=solution[0], him=solution[1],
                            nf=solution[2], mf=solution[3])

        sqr_sum = 0
        for a, b in zip(y_data, output):
            sqr_sum += (a-b)**2
        n_rmse = np.sqrt(np.mean(sqr_sum))/np.mean(y_data)
        return -n_rmse


    def on_gen(ga_instance):
        print(f"Generation = {ga_instance.generations_completed}")
        print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")

    num_generations = 20  # Number of generations.
    num_parents_mating = 8

    sol_per_pop = 50  # Number of solutions in the population.
    num_genes = 4

    ga_instance = pg.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        mutation_num_genes=1,
                        fitness_func=fitness,
                        gene_space=[{"low": bounds[0][0], "high": bounds[1][0]},
                                    {"low": bounds[0][1], "high": bounds[1][1]},
                                    {"low": bounds[0][2], "high": bounds[1][2]},
                                    {"low": bounds[0][3], "high": bounds[1][3]}],
                        on_generation=on_gen)

    ga_instance.run()
    solution, solution_fitness, solution_idx =\
        ga_instance.best_solution(ga_instance.last_generation_fitness)

    return solution, model_func(angles=x_data,
                                hf=solution[0], him=solution[1],
                                nf=solution[2], mf=solution[3])
# ------------- Fitting tools ------------


# ------------ Script ------------
# curve = read_curve_data(file_name='T2 TE norm_bg_corr_peaks.csv')
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
