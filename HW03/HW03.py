### Daniel Kronovet (dbk2123)
### COMS W4721 HW 03
### March 31, 2015

'''
from HW03 import *
import core.classifiers as c
'''

import os

import matplotlib.pyplot as plt
import pandas as pd

from core import classifiers
from core import ensemble
from core import visualizer

path = os.path.dirname(os.path.realpath(__file__))

X = pd.read_csv(path + '/cancer_csv/X.csv', header=None)
y = pd.read_csv(path + '/cancer_csv/y.csv', header=None)

def part1():
    dist = pd.Series([0.1, 0.2, 0.3, 0.4], index=[1,2,3,4])
    n_values = [100, 200, 300, 400, 500]
    bootstrapper = ensemble.Bootstrapper()
    samples = {n: bootstrapper.sample_discrete(dist, n) for n in n_values}
    visualizer.plot_params_hist(samples)

if __name__ == '__main__':
    part1()


