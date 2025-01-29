from Nsga2 import *
from Clustering import *
import Global as Gl

#Boobs

if __name__ == '__main__':
    filename = ['Spheres 2 2000 10.csv', 'Rings 1500 3.csv']
    save_name = ['Spheres', 'Rings']
    ndata = [2000, 1500]
    num_real_clusters = [10]
    population_size = [100, 100]
    mock_Kmax = [20, 6]

    np.random.seed(26469)
    for i in range(len(filename)):

        # problem info
        filename_i = 'Datasets/' + filename[i]
        print('#########')
        print(str(i) + ' CURRENT DATASET: ' + filename_i + '')
        print('#########')

        data_info = {
            'ndata': ndata[i],
            'mdim': 2,
            'is_label': True,
            'num_real_clusters': num_real_clusters[i]
        }
        settings = {
            'distance': 'euclidean',
            'normalize': True,
            'separator': ','
        }
        Gl.problem = Problem(filename_i, data_info, settings)

        # nsga2 info
        Gl.population_size = population_size[i]
        Gl.mock_L = 10
        Gl.mock_Kmax = mock_Kmax[i]
        Gl.num_objectives = 2

        parameters = {
            'population_size': Gl.population_size,
            'crossover_prob': None,
            'mutation_prob': None,
            'representation': None,
            'max_eval': 10000,
            'filename': save_name[i]
        }

        # nsga2 run and get results
        alg = Nsga2(parameters, verbose = 1)
        alg.run()
        alg.output_generation()