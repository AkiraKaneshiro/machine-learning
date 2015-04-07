### Daniel Kronovet (dbk2123)
### COMS W4721 HW 04
### April 14, 2015

'''
from HW04.HW04 import *
'''

import os
import random
random.seed('Siddhartha')

import matplotlib.pyplot as plt
import pandas as pd

from core.ensemble import Bootstrapper

PATH = os.path.dirname(os.path.realpath(__file__))
ratings = pd.read_csv(PATH + '/movies_csv/ratings.txt', header=None)
ratings_test = pd.read_csv(PATH + '/movies_csv/ratings_test.txt', header=None)
movies = pd.read_csv(PATH + '/movies_csv/movies.txt', header=None)


def generate_sample():
    w = pd.Series([.2, .5, .3])
    n = 500
    bootstrapper = Bootstrapper(w, n)


def problem1():
    pass


if __name__ == '__main__':
    problem1()