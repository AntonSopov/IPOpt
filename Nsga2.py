# В ПРОЦЕССЕ

import numpy as np
import pandas as pd

import Global as Gl
import Population
from Problem import *

# binary nsga2
class Nsga2:
    verbose = False

    # genetic operators & evaluator
    initialization = None
    crossover_operator = None
    mutation_operator = None
    _evaluator = None               # collection of performance measures and evaluation functions, Evaluator

    # nsga2 & parameters
    # ***
    # ***
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

        # пройтись по параметрам

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