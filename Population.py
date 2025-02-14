'''
Population.py
Включает описание класса Population, который 

Методы:
__init__: конструктор класса; принимает 


'''

import numpy as np
from Solution import *

class Population:
    _solutions = None
    _size = None
    _max_size = None

    def __init__(self, max_size):
        self._size = 0
        self._max_size = max_size
        self._solutions = [None] * self._max_size
    
    def add(self, solution):
        if self._max_size <= self._size:
            print('Cannot add more solutions. Maximum capacity is reached!')
        else:
            self._solutions[self._size] = solution
            self._size += 1
    
    def clear(self):
        self._size = 0
        self._solutions = [None] * self._max_size

    

    # getter functions
    def __getitem__(self, i): return self._solutions[i]

    def size(self): return self._size