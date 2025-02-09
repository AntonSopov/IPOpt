'''
Problem.py
Включает описание класса Problem, который автоматически формирует задачу нахождения оптимального портфеля инвестиций. ПОКА НЕ ТЕСТИРОВАЛОСЬ

Методы:
__init__: конструктор класса; принимает на вход файлы и приводит данные из них в вид, удобный для дальнейшей работы. нужный вид файла потом загружу
set_parameters: опционально, указывает дополнительные ограничения (максимальные риск и прибыль). определяет число ограничений с учётом опциональных
!! optimize: находит набор оптимальных портфелей с помощью выбранного метода

objIncome: целевая функция 1 (прибыль). возвращает вещественное значение
objRisk: целевая функция 2 (риск). возвращает вещественное значение
constraints: проверка ограничений (<=). возвращает массив вещественных значений (разность между левой и правой частями). если число превышает 0 - это ограничение не выполняется
'''

import numpy as np
import pandas as pd

import Nsga2

class Problem:
    # region parameters
    data = None                 #  list of data files
    filename = None             #  list of string, data filenames
    project_names = None        #  list of lists of string, project names per subdivision

    # project parameters
    inc = None                  # П_ij
    risk = None                 # R_ij
    cost = None                 # b_ij
    N = None                    # N_i
    C = None                    # C

    # problem input parameters
    budget = None               # B_i
    overallBudget = None        # B_C
    investmentBudget = None     # B_E
    maxIncome = None            # I (not stated in colab)
    maxRisk = None              # p

    # logs and reports
    verbose = None

    # optimization
    num_of_constraints = None

    # endregion parameters


    # region init
    def __init__(self, filenames, separator, verbose = 0):
        self.verbose = verbose

        self.filename = filenames
        self.C = len(filenames)
        self.N = np.empty(self.C)

        # data loading
        self.budget = np.empty(self.C)

        for i in range(self.C):
            # reading budget
            self.budget[i] = pd.read_csv(filenames[i], sep = separator, header = None, nrows = 1).to_numpy()[0][0]  # that's an ugly way to read a number from the first row... but it works!

            # data
            self.data[i] = pd.read_csv(filenames[i], sep = separator, skiprows = [0])
            self.N[i] = self.data[i].shape[0]

        # extracting everything from data
        self.project_names = []
        self.inc = []
        self.risk = []
        self.cost = []

        for i in range(self.C):
            if 'Название' in self.data.columns:
                self.project_names.append(self.data['Название'].to_list())
            self.inc[i].append( self.data['Прибыль'].to_numpy() )
            self.risk[i].append( self.data['Риск'].to_numpy() )
            self.cost[i].append( self.data['Стоимость'].to_numpy() )
        
        # additional calculations
        self.overallBudget = np.sum(self.budget)
    # endregion init

    # region set_parameters
    def set_parameters(self, maxIncome = None, maxRisk = None):
        self.maxIncome = maxIncome
        self.maxRisk = maxRisk

        if maxIncome == None and maxRisk == None:
            self.num_of_constraints = self.C + 2
        elif maxIncome == None or maxRisk == None:
            self.num_of_constraints = self.C + 1
        else:
            self.num_of_constraints = self.C
    # endregion set_parameters

    def optimize(self, method = 'nsga2'):
        1




    # region objectives
    # first objective: Income -> max
    def objIncome(self, x):
        income = 0
        for i in range(self.C):
            for j in range(self.N[i]):
                income += self.inc[i][j] * x[i][j]
        
        return income

    # second objective: Risk -> min
    def objRisk(self, x):
        r = 1. / np.sum(x)
        for i in range(self.C):
            for j in range(self.N[i]):
                r += self.risk[i][j] * x[i][j]
        
        return r
    
    # endregion objectives

    # region constraints
    # all of them are inequality constraints
    def constraints(self, x):
        constr = np.empty(self.num_of_constraints)

        # first constraint: overall cost does not exceed the overall budget + investment budget
        constr[0] = overall_summ - (self.overallBudget - self.investmentBudget)
        
        # next i constraints: cost of the projects per each subdivision does not exceed the budget
        overall_summ = 0
        for i in range(self.C):
            summ = 0
            for j in range(self.N[i]):
                summ += self.cost[i][j] * x[i][j]
            constr[i + 1] = summ - self.budget[i]
            overall_summ += summ
        
        # Last two optional constraints: Overall Income <= Max Income  &&  Overall Risk <= Max Risk
        if self.maxIncome != None and self.maxRisk != None:
            constr[-2] = self.objIncome(x) - self.maxIncome
            constr[-1] = self.objRisk(x) - self.maxRisk
        elif self.maxIncome != None:
            constr[-1] = self.objIncome(x) - self.maxIncome
        elif self.maxRisk != None:
            constr[-1] = self.objRisk(x) - self.maxRisk

        return constr