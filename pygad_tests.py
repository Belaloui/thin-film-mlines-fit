# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:11:26 2022
"""

import pygad as pg
import numpy as np
from random import random
import matplotlib.pyplot as plt


# -------- Data generation and model ----------- #
def model_func(x, alpha, beta):
    return alpha*x + beta


ALPHA = 3.5
BETA = 5

x_data = np.linspace(0, 10, 20)
y_data = [model_func(x, ALPHA, BETA) + (6*random()-3) for x in x_data]
y_data_exact = [ALPHA*x + BETA for x in x_data]
# -------- Data generation and model ----------- #


# -------- Genetic Algorithm ----------- #
def fitness(solution, solution_idx):
    global y_data
    output = model_func(x=x_data, alpha=solution[0], beta=solution[1])

    sqr_sum = 0
    for a, b in zip(y_data, output):
        sqr_sum += (a-b)**2
    n_rmse = np.sqrt(np.mean(sqr_sum))/np.mean(y_data)
    return -n_rmse


num_generations = 100  # Number of generations.
num_parents_mating = 20

sol_per_pop = 100  # Number of solutions in the population.
num_genes = 2

ga_instance = pg.GA(num_generations=num_generations,
                    num_parents_mating=num_parents_mating,
                    sol_per_pop=sol_per_pop,
                    num_genes=num_genes,
                    mutation_num_genes=1,
                    fitness_func=fitness,
                    gene_space=[{"low": 0, "high": 8},
                                {"low": 0, "high": 10}])

ga_instance.run()
solution, solution_fitness, solution_idx =\
    ga_instance.best_solution(ga_instance.last_generation_fitness)

print('\nSolution =', solution)

plt.plot(x_data, y_data, linestyle="None", marker='*', label='data')
plt.plot(x_data, y_data_exact, label='Exact model')
plt.plot(x_data, model_func(x_data, solution[0], solution[1]),
         label='Fitted model')
plt.legend()
plt.show()

# -------- Genetic Algorithm ----------- #
