### Daniel Kronovet (dbk2123)
### COMS W4721 HW 03
### March 31, 2015

'''
from HW03.HW03 import *
import core.classifiers as c
logit = c.BinaryLogit(X_train, y_train, X_test, y_test)
bayes = c.BinaryBayes(X_train, y_train, X_test, y_test)

import core.ensemble as e
adaboost = e.AdaBoost(X_train, y_train, X_test, y_test, c.BinaryBayes)
'''

import os

import matplotlib.pyplot as plt
import pandas as pd

from core.classifiers import BinaryBayes, BinaryLogit, OnlineBinaryLogit
from core.ensemble import AdaBoost, Bootstrapper
from core import visualizer

PATH = os.path.dirname(os.path.realpath(__file__))
X = pd.read_csv(PATH + '/cancer_csv/X.csv', header=None)
y = pd.read_csv(PATH + '/cancer_csv/y.csv', header=None)

# X_test, X_train = X.ix[499:], X.ix[:499]
X_test, X_train = X.ix[:183], X.ix[183:]
# y_test, y_train = y.ix[499:], y.ix[:499]
y_test, y_train = y.ix[:183], y.ix[183:]

X_train.index = range(len(X_train))
y_train.index = range(len(y_train))
X_test.index = range(len(X_test))
y_test.index = range(len(y_test))
assert len(X_train) == 500, 'Train set != 500.'

def part1():
    dist = pd.Series([0.1, 0.2, 0.3, 0.4], index=[1,2,3,4])
    n_values = [100, 200, 300, 400, 500]
    bootstrapper = Bootstrapper()
    samples = {n: bootstrapper.sample_discrete(dist, n) for n in n_values}
    visualizer.plot_params_hist(samples)

def part2():
    adaboost = AdaBoost(X_train, y_train, X_test, y_test, BinaryBayes)
    adaboost.iterate(10)
    return adaboost
    # w indices: 100, 280, 420

def part3():
    adaboost = AdaBoost(X_train, y_train, X_test, y_test, OnlineBinaryLogit)
    adaboost.iterate(10)
    return adaboost

if __name__ == '__main__':
    part1()
    part2()
    part3()


