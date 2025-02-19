from Nsga2 import *
from Problem import *
import Global as Gl

if __name__ == '__main__':
    np.random.seed(3586)
    # initializing the problem
    filenames = ['datasets\krskmz\cfo1.csv', 'datasets\krskmz\cfo2.csv', 
                'datasets\krskmz\cfo3.csv', 'datasets\krskmz\cfo4.csv', 
                'datasets\krskmz\cfo5.csv']
    test_problem = Problem(filenames, ',')
    test_problem.set_parameters(investmentBudget = 40.0, maxRisk = 2.0)

    # setting global variables
    Gl.problem = test_problem
    Gl.num_objectives = 2
    Gl.population_size = 60

    opt_params = {
        'population_size': Gl.population_size,
        'crossover_prob': 1.0, 
        'mutation_prob': 1.0,
        'max_eval': 1000,
        'constraints': True,
        'eps': 0,
        'filename': 'test01'
    }
    Gl.problem.optimize('nsga2', opt_params)