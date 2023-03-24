from reflection_coeffs import ReflectionModel
import numpy as np
import pygad as pg

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