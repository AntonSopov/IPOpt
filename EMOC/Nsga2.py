import numpy as np
import pandas as pd

import Global as Gl
from Population import *
from Solution import *
from Evaluator import *
from Clustering import *
from Problem import *


class Nsga2:
    verbose = False

    # optimizer parameters
    _evaluator = None          # collection of performance measures and evaluation functions, Evaluator

    # genetic operators & evaluator
    initialization = None
    crossover_operator = None
    mutation_operator = None
    _evaluator = None

    # nsga2 & clustering parameters
    _population_size = None     # int
    _crossover_prob = None      # double, [0, 1]
    _mutation_prob = None       # double, [0, 1]
        
    _representation = None      # string // binary, locus

    _max_solutions = None       # int

    # population info
    _population = None          # Population, main (parent) population
    _offspring = None           # Population, offspring population
    _auxiliary = None           # Population, auxiliary population (for replacement strategy) 

    # iteration info
    _generation = None
    _FEs = None

    # stopping criteria info
    _max_generations = None     # not used
    _max_eval = None

    # auxiliary structures
    _nds_front = None           # np.array(2d), non-dominating sorting fronts (first column is a counter)
    _nds_s = None               # np.array(2d), for nds (the list of individuals that are dominated) (first column is a counter)
    _nds_n = None               # np.array(1d), for nds (individual rank)
    _idx_val_tuples = None      # np.array(1d), for crowding distance
    _parents = None             # np.array(1d), for mating selection

    # logs
    _filename = None


    def __init__(self, parameters, verbose = 0):
        self.verbose = verbose

        if parameters['population_size'] != None:
            self._population_size = parameters['population_size']
        else:
            self._population_size = 100

        if parameters['crossover_prob'] != None:
            self._crossover_prob = parameters['crossover_prob']
        else:
            self._crossover_prob = 1.0

        if parameters['mutation_prob'] != None:
            self._mutation_prob = parameters['mutation_prob']
        else:
            self._mutation_prob = 1.0

        if parameters['representation'] != None:
            self._representation = parameters['representation']
        else:
            self._representation = 'locus'

        if parameters['max_eval'] != None:
            self._max_eval = parameters['max_eval']
        else:
            self._max_eval = 5000

        if parameters['filename'] != None:
            self._filename = parameters['filename']
        else:
            self._filename = self._population_size

        self._configure()

    def _configure(self):
        # determining maximum number of solutions
        self._max_solutions = 2 * self._population_size

        # creating population instances
        if self._population_size % 4 == 0:
            self._population = Population(self._population_size)
            self._offspring = Population(self._population_size)
            self._auxiliary = Population(self._max_solutions)
        else:
            raise ValueError('Population size needs to be a multiple of 4')

        # creating auxiliary data structures
        self._nds_front = np.zeros( (self._max_solutions, self._max_solutions + 1), dtype = int )
        self._nds_s = np.zeros( (self._max_solutions, self._max_solutions + 1), dtype = int )
        self._nds_n = np.zeros( self._max_solutions, dtype = int )
        self._idx_val_tuples = np.empty( (self._max_solutions, 2), dtype = float )
        self._parents = np.arange( 0, self._population_size, dtype = int )

        # setting genetic operators depending on encoding type
        if self._representation == "locus":
            self.initialization = self.hybrid_initialization
            self.crossover_operator = self.uniform_crossover
            self.mutation_operator = self.neighbourhood_biased_mutation
            self._evaluator = Evaluator()
        
        '''# setting stopping criterion
        self._max_eval = ( self._population_size * self._max_generations )'''
    
    def run(self):
        if self.verbose: print('Initializing the algorithm...')

        self.initialization()
        if self.verbose: print('Initialized succesfully!')
        self.generate_evaluate_initial_population()
        if self.verbose: print('Evaluated initial population succesfully!')

        self._generation = 1
        if self.verbose: print('Starting main cycle...')
        while self._evaluator.total_evaluations() <= self._max_eval:
            if self.verbose: print('Generation:', self._generation)
            # selection, crossover, mutation
            self.genetic_operators()

            # evaluating offspring
            self._evaluator.evaluate_pop(self._offspring)

            # solution replacement
            self.survival_selection()

            self._generation += 1

        if self.verbose: print('Finished!')
        if self.verbose: print('Saving the result...')

    # for hybrid initialization
    def create_priority_solution(self, n):
        # creating new mst solution
        sol = Solution(enc = None, init = False)
        for i in range(Gl.problem.ndata()):
            sol[i] = Gl.problem.mst_edge(i)
        
        # removing n highest priority links
        for i in range(n):
            edge = Gl.problem.priority_edge(i)
            #sol[edge] = sol.random_encoding(edge, sol[edge])
            sol[edge] = sol.random_encoding_init(sol.encoding(), edge, sol[edge]) # replacing this edge with one of nn (but not itself, 2nd arg)

        return sol             

    def hybrid_initialization(self):
        # adding full mst solution
        print('full mst solution')
        sol = Solution(enc = None, init = False)
        for i in range(Gl.problem.ndata()):
            sol[i] = Gl.problem.mst_edge(i)
        self._auxiliary.add(sol)

        # filling auxiliary population with interesting solutions
        possible_k = np.arange(2, Gl.mock_Kmax + 1)

        while self._auxiliary.size() < self._max_solutions:
            np.random.shuffle(possible_k)
            for k in range(2, Gl.mock_Kmax):
                # current k
                target_k = possible_k[k - 2]

                # priority solution using selected k
                new_sol = self.create_priority_solution(target_k - 1)
                self._auxiliary.add(new_sol)
                new_sol = None

                if self._auxiliary.size() == self._max_solutions:
                    break
        
        # initializing offspring population
        for i in range(self._population_size):
            solut = Solution(enc = None, init = False)
            self._offspring.add(solut)
        
    
    def generate_evaluate_initial_population(self):
        # evaluating initial solutions
        self._evaluator.evaluate_pop(self._auxiliary)

        # population of nonselected individuals
        pop_to_remove = Population(self._auxiliary.size() - self._population_size)

        # replacement
        self._population.clear()
        self.replacement_strategy(self._auxiliary, self._population, pop_to_remove)
        self._auxiliary.clear()
        pop_to_remove.clear()



    # TODO: neighbourhood_biased_mutation
    def genetic_operators(self):
        # for every 4 parents, 2 child is produced => two iterations are needed
        ctr = 0
        for rep in range(2):
            # permutation of the parent indexes
            np.random.shuffle(self._parents)

            # applying genetic operators
            for i in range(0, self._population_size, 4):
                # selection (binary tournament)
                parent1 = self.binary_tournament(self._population[self._parents[i]], self._population[self._parents[i + 1]])
                parent2 = self.binary_tournament(self._population[self._parents[i + 2]], self._population[self._parents[i + 3]])

                # crossover
                self.crossover_operator(parent1, parent2, self._offspring[ctr], self._offspring[ctr + 1], self._crossover_prob)
                
                # mutation
                self.mutation_operator(self._offspring[ctr], self._mutation_prob)
                self.mutation_operator(self._offspring[ctr + 1], self._mutation_prob)

                ctr += 2
    
    def binary_tournament(self, s1, s2):
        # dominance comparison
        dominance = self.pareto_dominance(s1.objective(), s2.objective())
        if dominance == 1:
            return s1
        if dominance == 2:
            return s2
        
        # crowding distance comparison
        if s1.crowding_distance() > s2.crowding_distance():
            return s1
        if s1.crowding_distance() < s2.crowding_distance():
            return s2
        
        # if dominance and crowding distance values are equal, make random decision
        rand = np.random.rand()
        if rand < 0.5:
            return s1
        else:
            return s2
    
    def uniform_crossover(self, parent1, parent2, child1, child2, crossover_prob):
        rand = np.random.rand()
        if rand < crossover_prob:
            for allele in range(parent1.encoding_length()):
                if np.random.rand() < 0.5:
                    child1[allele] = parent1[allele]
                    child2[allele] = parent2[allele]
                else:
                    child1[allele] = parent2[allele]
                    child2[allele] = parent1[allele]
        else:
            for allele in range(parent1.encoding_length()):
                child1[allele] = parent1[allele]
                child2[allele] = parent2[allele]
                
    def neighbourhood_biased_mutation(self, solution, mutation_prob):
        enc_length = solution.encoding_length()
        prob = mutation_prob / enc_length

        # mutating each allele
        for i in range(enc_length):
            # ranking current link
            rank = Gl.problem.neighbour_rank(i, solution[i])
            # calculating the probability of mutating the link i -> solution[i]
            allele_prob = prob + (rank / enc_length)**2

            # mutating allele
            if np.random.rand() < allele_prob:
                # replacing solution[i] with random alternative
                solution[i] = solution.random_encoding(i, solution[i])



    # p, q - objective vectors
    def pareto_dominance(self, p, q):
        ctr_p, ctr_q, ctr_eq = 0, 0, 0
        size = len(p)

        for i in range(size):
            if p[i] < q[i]:
                ctr_p += 1
            elif p[i] > q[i]:
                ctr_q += 1
            else:
                ctr_eq += 1
        
        if (ctr_q == 0) and (ctr_p > 0):
            return 1        # p dominates q
        if (ctr_p == 0) and (ctr_q > 0):
            return 2        # q dominates p
        if ctr_eq == size:
            return 3        # p and q are equal
        
        return 0            # or they are incomparable

    def nondominated_sorting(self, population):
        # getting population size
        size = population.size()

        # initializing
        for p in range(size):
            self._nds_front[p][0] = 0
            self._nds_s[p][0] = 0
            self._nds_n[p] = 0

        # first front
        for p in range(size):
            for q in range(size):
                if p != q:
                     # 1 dominates, 2 is dominated, 3 equal, 0 incomparable
                    dominance = self.pareto_dominance(population[p].objective(), population[q].objective())
                    
                    # if p dominates q => q goes in _nds_s[p]
                    if dominance == 1:
                        self._nds_s[p][0] += 1
                        self._nds_s[p][self._nds_s[p][0]] = q
                    
                    # if q dominates p => p rank should be increased
                    elif dominance == 2:
                        self._nds_n[p] += 1
                
            # if the individual remained nondominated, it should be in the first front
            if self._nds_n[p] == 0:
                self._nds_front[0][0] += 1
                self._nds_front[0][self._nds_front[0][0]] = p
                population[p].set_rank(1)
        
        # other fronts
        # i - front, p - 1st solution, q - 2nd solution
        for i in range(size - 1):
            if self._nds_front[ i ][ 0 ] == 0:
                break

            self._nds_front[i + 1][0] = 0

            for p in range(1, self._nds_front[i][0] + 1):
                for q in range(1, self._nds_s[self._nds_front[i][p]][0] + 1):
                    self._nds_n[ self._nds_s[self._nds_front[i][p]][q] ] -= 1

                    # if the individual remained nondominated, it should be in the (i + 2) front
                    if self._nds_n[ self._nds_s[ self._nds_front[i][p] ][q] ] == 0:
                        self._nds_front[i + 1][0] += 1
                        self._nds_front[i + 1][self._nds_front[i + 1][0]] = self._nds_s[self._nds_front[i][p]][q]
                        population[ self._nds_s[self._nds_front[i][p]] [q] ].set_rank(i + 2)


    def crowding_distance(self, population, front):
        # initializing crowding distance for each individual in population
        for i in range(1, front[0] + 1):
            population[front[i]].set_crowding_distance(0)
        
        # computing crowding distance
        for m in range(Gl.num_objectives):
            for i in range(1, front[0] + 1):
                tuples = self._idx_val_tuples[:front[0]]

                # inserting index
                tuples[i - 1][0] = front[i]
                # inserting current objective
                tuples[i - 1][1] = population[front[i]].objective(m)
            
            # sorting tuples in ascending order by values
            tuples = tuples[tuples[:, 1].argsort()[::1]]

            if tuples[0][1] != tuples[-1][1]:
                # best and worst solution per objective gets np.inf
                population[int(tuples[0][0])].set_crowding_distance(np.inf)
                population[int(tuples[-1][0])].set_crowding_distance(np.inf)
                # for the rest of solutions, calculating distance
                for i in range(1, front[0] - 1):                    # excluding best and worst
                    #if population[self._idx_val_tuples[i][0]].crowding_distance() != np.inf:  нужно ли это...

                    # считается так: crwd += (crwd[i + 1] - crwd[i - 1]) / (max crwd - min crwd)
                    crwd = population[int(tuples[i][0])].crowding_distance()
                    crwd += ( tuples[i + 1][1] - tuples[i - 1][1] ) / ( tuples[-1][1] - tuples[0][1] )
                    population[int(tuples[i][0])].set_crowding_distance(crwd)




    def replacement_strategy(self, source, selected, nonselected):
        self.nondominated_sorting(source)
        
        f = 0       # current front
        while selected.size() < self._population_size:
            # computing crowding distance of individuals in the current front
            self.crowding_distance(source, self._nds_front[f])
            
            if (selected.size() + self._nds_front[f][0]) <= self._population_size:
                # adding individuals to the current front if there is available space
                for i in range(1, self._nds_front[f][0] + 1):
                    selected.add(source[self._nds_front[f][i]])
            
            else:
                # ...otherwise, using crowding distance
                #print(self._nds_front[f][0])
                tuples = self._idx_val_tuples[:self._nds_front[f][0]]
                for i in range(1, self._nds_front[f][0] + 1):
                    # inserting indexes
                    tuples[i - 1][0] = self._nds_front[f][i]

                    # inserting crowding distance values
                    tuples[i - 1][1] = source[self._nds_front[f][i]].crowding_distance()
                
                # sorting tuples in descending order of crowding distance
                tuples = tuples[tuples[:, 1].argsort()[::-1]]

                # filling population with individual from less crowded regions
                i = 0
                while selected.size() < self._population_size:
                    selected.add(source[int(tuples[i][0])])
                    i += 1
                
                # adding the rest of the individuals to the nonselected (children) population
                while i < self._nds_front[f][0]:
                    nonselected.add(source[int(tuples[i][0])])
                    i += 1
            
            f += 1
        
        # remaining individuals are also copied to children population
        while (selected.size() + nonselected.size()) < source.size():
            for t in range(1, self._nds_front[f][0] + 1):
                nonselected.add(source[self._nds_front[f][t]])
            f += 1

    def survival_selection(self):
        # merging parent and offspring populations
        self._auxiliary.clear()     # на всякий случай
        for i in range(self._population_size):
            self._auxiliary.add(self._population[i])
            self._auxiliary.add(self._offspring[i])
        
        # emptying populations
        self._population.clear()
        self._offspring.clear()

        # applying replacement strategy to form parent and offspring populations
        # based on nondominated sorting and crowding distance
        self.replacement_strategy(self._auxiliary, self._population, self._offspring)

        # emptying auxiliary population
        self._auxiliary.clear()
    


    # results function
    def output_generation(self):
        self.nondominated_sorting(self._population)

        # initializing dataframes 
        df_measures = pd.DataFrame()
        df_cluster = pd.DataFrame()

        # initializing measure lists
        variance, connectivity, k, ari = [], [], [], []

        ### Measures file
        ctr = 0     # how many solutions in the pareto front
        for i in range(self._population_size):
            if self._population[i].rank() == 1:
                ctr += 1
                
                # decoding current solution
                clustering = self._population[i].decode_clustering()

                # calculating intra-cluster variance, connectivity and number of clusters
                variance.append(self._evaluator.variance(clustering))
                connectivity.append(self._evaluator.connectivity(clustering))
                k.append(clustering.total_clusters())

                # calculating adjusted rand index if possible
                if Gl.problem.is_label():
                    ari.append(self._evaluator.adjusted_rand_index(clustering))

            ### Cluster file (current solution)
            labels_current = []
            for j in range(Gl.problem.ndata()):
                labels_current.append(clustering.assignment(j))
            df_cluster[str(ctr)] = labels_current
        
        # finalizing measures dataframe
        df_measures['variance'] = variance
        df_measures['connectivity'] = connectivity
        df_measures['k'] = k
        if Gl.problem.is_label():
            df_measures['ari'] = ari
        
        # saving dataframes
        df_measures.to_csv(f'{self._filename}_measures.csv')
        df_cluster.to_csv(f'{self._filename}_clustering.csv')