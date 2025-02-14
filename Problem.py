'''
Problem.py
Включает описание класса Problem, который автоматически формирует задачу нахождения оптимального портфеля инвестиций. ПОКА НЕ ТЕСТИРОВАЛОСЬ

Методы:
__init__: конструктор класса; принимает на вход файлы и приводит данные из них в вид, удобный для дальнейшей работы. нужный вид файла потом загружу
set_parameters: опционально, указывает дополнительные ограничения (максимальные риск и прибыль). определяет число ограничений с учётом опциональных
!! optimize: находит набор оптимальных портфелей с помощью выбранного метода

objIncome: целевая функция 1 (прибыль). возвращает вещественное значение
objRisk: целевая функция 2 (риск). возвращает вещественное значение
constraints: проверка ограничений (<=). возвращает массив вещественных значений (разность между левой и правой частями) и допустимость решения. если число превышает 0 - это ограничение не выполняется

evaluate: обновляет целевую функцию и ограничения у индивида (Solution)
evaluate_pop: делает это для всей популяции (Population)
'''

import numpy as np
import pandas as pd

import Nsga2

class Problem:
    # region parameters
    _data = None                 #  list of data files
    _filename = None             #  list of string, data filenames
    _project_names = None        #  list of lists of string, project names per subdivision

    # project parameters
    _inc = None                  # П_ij
    _risk = None                 # R_ij
    _cost = None                 # b_ij
    _N = None                    # N_i
    _C = None                    # C
    _encoding_length = None      # the sum of N_i

    # problem input parameters
    _budget = None               # B_i
    _overallBudget = None        # B_C
    _investmentBudget = None     # B_E
    _maxIncome = None            # I (not stated in colab)
    _maxRisk = None              # p

    # logs and reports
    _verbose = None

    # optimization
    _num_of_constraints = None
    _total_evaluations = None

    # endregion parameters


    # region init
    def __init__(self, filenames, separator, verbose = 0):
        self._verbose = verbose

        self._filename = filenames
        self._C = len(filenames)
        self._N = np.empty(self._C)

        # data loading
        self._budget = np.empty(self._C)

        for i in range(self._C):
            # reading budget
            self._budget[i] = pd.read_csv(filenames[i], sep = separator, header = None, nrows = 1).to_numpy()[0][0]  # that's an ugly way to read a number from the first row... but it works!

            # data
            self._data[i] = pd.read_csv(filenames[i], sep = separator, skiprows = [0])
            self._N[i] = self._data[i].shape[0]

        # extracting everything from data
        self._project_names = []
        self._inc = []
        self._risk = []
        self._cost = []

        for i in range(self._C):
            if 'Название' in self._data.columns:
                self._project_names.append(self._data['Название'].to_list())
            self._inc[i].append( self._data['Прибыль'].to_numpy() )
            self._risk[i].append( self._data['Риск'].to_numpy() )
            self._cost[i].append( self._data['Стоимость'].to_numpy() )
        
        # additional calculations
        self._overallBudget = np.sum(self._budget)
        self._encoding_length = np.sum(self._N)
    # endregion init

    # region set_parameters
    def set_parameters(self, maxIncome = None, maxRisk = None):
        self._maxIncome = maxIncome
        self._maxRisk = maxRisk

        if maxIncome == None and maxRisk == None:
            self._num_of_constraints = self._C + 2
        elif maxIncome == None or maxRisk == None:
            self._num_of_constraints = self._C + 1
        else:
            self._num_of_constraints = self._C
    # endregion set_parameters

    def optimize(self, method = 'nsga2'):
        1




    # region objectives
    # first objective: Income -> max
    def objIncome(self, x):
        income = 0
        for i in range(self._C):
            for j in range(self._N[i]):
                income += self._inc[i][j] * x[i][j]
        
        return -income

    # second objective: Risk -> min
    def objRisk(self, x):
        r = 1. / np.sum(x)
        for i in range(self._C):
            for j in range(self._N[i]):
                r += self._risk[i][j] * x[i][j]
        
        return r
    
    # endregion objectives

    # region constraints
    # all of them are inequality constraints
    def constraints(self, x):
        constr = np.empty(self._num_of_constraints)
        
        # second i constraints: cost of the projects per each subdivision does not exceed the budget
        overall_summ = 0
        for i in range(self._C):
            summ = 0
            for j in range(self._N[i]):
                summ += self._cost[i][j] * x[i][j]
            constr[i + 1] = summ - self._budget[i]
            overall_summ += summ
        
        # first constraint: overall cost does not exceed the overall budget + investment budget
        constr[0] = overall_summ - (self._overallBudget - self._investmentBudget)

        # Last two optional constraints: Overall Income <= Max Income  &&  Overall Risk <= Max Risk
        if self._maxIncome != None and self._maxRisk != None:
            constr[-2] = self.objIncome(x) - self._maxIncome
            constr[-1] = self.objRisk(x) - self._maxRisk
        elif self._maxIncome != None:
            constr[-1] = self.objIncome(x) - self._maxIncome
        elif self._maxRisk != None:
            constr[-1] = self.objRisk(x) - self._maxRisk

        # is x feasible
        feasibility = True
        for i in range(self._num_of_constraints):
            if constr[i] > 0:
                feasibility = False
                break

        return constr, feasibility
    
    # region evaluate functions
    # evaluate // encoding
    def evaluate(self, solution):
        self._total_evaluations += 1
        enc = solution.encoding()

        # evaluating objectives and constraints, saving information in the solution object
        obj1 = self.objIncome(enc)
        solution.set_objective(0, obj1)

        obj2 = self.objRisk(enc)
        solution.set_objective(1, obj2)

        constr, feasibility = self.constraints(enc)
        solution.set_constraints(constr)
        solution.set_feaibility(feasibility)

        solution.set_eval(self._total_evaluations)
    
    # evaluate // population
    def evaluate_pop(self, population):
        for i in range(population.size()):
            self.evaluate(population[i])
    
    # endregion evaluate functions



    # getter functions
    def encoding_length(self): return self._encoding_length
    def num_constraints(self): return self._C