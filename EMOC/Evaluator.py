import numpy as np
import math
import Global as Gl
from Clustering import *
from Population import *
from Solution import *

class Evaluator:
    _total_evaluations = None       # int, evaluations counter
    _nn = None                      # int, number of nearest neighbours to use (_nn = mock_L)
    _connectivity_penalty = None    # np.array(1d, float), pre-computed penalties for connectivity measure

    def __init__(self):
        self._total_evaluations = 0
        self._nn = Gl.mock_L
        
        # precomputing connectivity penalties
        # Penalty = 1 / (Pos + 1), where Pos is the position in NN list
        self._connectivity_penalty = np.empty(self._nn)
        for i in range(self._nn):
            self._connectivity_penalty[i] = 1.0 / (i + 1.0)
        
    # evaluate // solution
    def evaluate(self, solution):
        self._total_evaluations += 1

        # decoding given solution into a clustering object
        clustering = solution.decode_clustering()

        # evaluating selected measures, saving information in the solution object
        solution_variance = self.variance(clustering)
        solution.set_objective(0, solution_variance)

        solution_connectivity = self.connectivity(clustering)
        solution.set_objective(1, solution_connectivity)

        solution.set_kclusters(clustering.total_clusters())
        solution.set_eval(self._total_evaluations)
    
    # evaluate // population
    def evaluate_pop(self, population):
        for i in range(population.size()):
            self.evaluate(population[i])
    

    ### INTERNAL CRITERIA ###
    def connectivity(self, clustering):
        conn = 0
        
        for i in range(Gl.problem.ndata()):
            # label of the current solution
            label = clustering.assignment(i)

            partial = 0
            for j in range(self._nn):
                # label of the j-th neighbour
                nn_label = clustering.assignment(Gl.problem.neighbour(i, j))

                if label != nn_label:
                    partial += self._connectivity_penalty[j]
                
            conn += partial
        
        return conn

    def variance(self, clustering):
        total_variance = 0

        for i in range(Gl.problem.ndata()):
            diff = pairwise_distances([Gl.problem.element(i)], [clustering.centre(clustering.assignment(i))], 
                                      metric = Gl.problem.distance_measure())[0]
            total_variance += diff**2
        
        return (total_variance / (Gl.problem.ndata()))[0]
    
    ### EXTERNAL CRITERIA ###   (for analysis)
    
    # ARI for Clustering object
    def adjusted_rand_index(self, clustering):
        # initializing
        cluster_size_real = np.zeros(Gl.problem.num_real_clusters(), dtype = int)
        cluster_size = np.zeros(clustering.total_clusters(), dtype = int)
        contingency  = np.zeros((clustering.total_clusters(), Gl.problem.num_real_clusters()), dtype = int)

        min_class_label = np.min(Gl.problem.label()) # min class label in data
        # contingency table
        for i in range(Gl.problem.ndata()):
            # obtained assignment for i-th element
            obtained = clustering.assignment(i)
            cluster_size[obtained] += 1

            # real assignment for i-th element
            real = Gl.problem.label(i) - min_class_label
            cluster_size_real[real] += 1

            # updating contingency table
            contingency[obtained][real] += 1
        
        # for ari equation (https://sci-hub.ru/10.1109/TEVC.2006.877146, p.68)
        sum_cells = 0
        sum_rows = 0
        sum_cols = 0

        for i in range(clustering.total_clusters()):
            sum_rows += combinations(cluster_size[i], 2)

            for j in range(Gl.problem.num_real_clusters()):
                sum_cells += combinations(contingency[i][j], 2)

                if i == 0:
                    sum_cols += combinations(cluster_size_real[j], 2)
        
        # numenator of the equation
        numenator = sum_cells - (sum_rows * sum_cols) / combinations(Gl.problem.ndata(), 2)

        # denominator of the equation
        denominator = 0.5 * (sum_rows + sum_cols) - (sum_rows * sum_cols) / combinations(Gl.problem.ndata(), 2)

        # calculating ari
        ari = numenator / denominator

        return ari
    
    # ARI for Solution object
    def adjusted_rand_index_sol(self, solution):
        clustering = solution.decode_clustering()
        ari = self.adjusted_rand_index(clustering)
        return ari
    
    # get_ and set_ functions
    def total_evaluations(self): return self._total_evaluations


# factorial between two values
def factorial(start, end = 2):
    sum = 1
    while start >= end:
        sum *= start
        start -= 1
    return sum

# combinations of n in k
def combinations(n, k):
    if k > n:
        return 0
    if k == n:
        return 1
    return factorial(n, n-k+1) / factorial(k)

