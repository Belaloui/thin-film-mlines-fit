from reflection_coeffs import ReflectionModel
import numpy as np
import pygad as pg

# ------------- Fitting tools ------------
class ModelFunction:
    """ A tool to get a model's function where certain parameters
    can be fixed.
    The fixed parameters are specified at intialization. The model's function
    to be fitted is simply the model_func method.
    """
    def __init__(self, polarization :str, fixed_params : dict):
        self.polarization = polarization
        self.fixed_params = fixed_params

    def model_func(self, *args):
        
        angles = args[0]
        params = [0, 0, 0, 0, 0, 0]
        variables = ['h_immers', 'h_film', 'n_substr', 'm_substr',
                     'n_film', 'm_film']
        
        arg_id = 1
        for ind, var in enumerate(variables):
            if var in self.fixed_params:
                params[ind] = self.fixed_params[var]
            else:
                params[ind] = args[arg_id]
                arg_id += 1

        model = ReflectionModel(lamb=632.8, n_prism=(2.5822, 2.8639),
                                h_immers=params[0],
                                h_film=params[1],
                                n_substr=params[2],
                                m_substr=params[3],
                                n_film=params[4],
                                m_film=params[5])
        if self.polarization == 's':
            ret = model.Rs_curve_fit(angles)
        elif self.polarization == 'p':
            ret = model.Rp_curve_fit(angles)
        return ret

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