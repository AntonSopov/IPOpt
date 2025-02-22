'''
ОПИСАНИЕ
'''

import numpy as np
import Global as Gl

class Solution:
    # attributes
    _encoding = None            # np.array(int), genotype
    _objective = None           # np.array(float), objective function values
    _constraints = None         # np.array(float), constraints values
    _feasibility = None         # bool, is feasible

    _eval = None                # int, the evaluation number at which the solution was found
    _rank = None                # int, rank (front) of the solution
    _crowding_distance = None   # float, crowding distance measure (for nsga2)

    # if enc != None: copy an individual
    # if enc == None and init == False: make an empty individual
    # if enc == None and init == True: randomly generate an individual
    def __init__(self, enc = None, init = True):
        self._objective = np.empty(Gl.num_objectives, dtype = float)

        self._eval = 0
        self._rank = 0
        self._crowding_distance = 0
        
        if enc != None:
            self._encoding = enc.copy()

        elif init == False:
            self._encoding = np.empty(self.encoding_length(), dtype = int)
            
        elif init == True:
            self._encoding = np.empty(self.encoding_length(), dtype = int)
            for i in range(self.encoding_length()):
                if np.random.uniform() <= 0.5:
                    self._encoding[i] = 0
                else:
                    self._encoding[i] = 1
    
    # the length of genotype is equal to the number of projects
    def encoding_length(self):
        return Gl.problem.encoding_length()



    # get_ and set_ functions
    def __getitem__(self, i): return self._encoding[i]
    def __setitem__(self, i, val): self._encoding[i] = val

    def encoding(self, i = None): return self._encoding if i == None else self._encoding[i]
    def set_encoding(self, i, val): self._encoding[i] = val
    def objective(self, i = None): return self._objective if i == None else self._objective[i]
    def set_objective(self, i, val): self._objective[i] = val
    def constraints(self, i = None): return self._constraints if i == None else self._constraints[i]
    def set_constraints(self, val): self._constraints = val
    def feasibility(self): return self._feasibility
    def set_feasibility(self, val): self._feasibility = val

    def eval(self): return self._eval
    def set_eval(self, val): self._eval = val
    def rank(self): return self._rank
    def set_rank(self, val): self._rank = val
    def crowding_distance(self): return self._crowding_distance
    def set_crowding_distance(self, val): self._crowding_distance = val