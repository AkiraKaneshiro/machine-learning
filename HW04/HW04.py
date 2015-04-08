### Daniel Kronovet (dbk2123)
### COMS W4721 HW 04
### April 14, 2015

'''
from HW04.HW04 import *
kms = problem1()

s = generate_sample()
'''

import os
import random
random.seed('Siddhartha')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from core.supervised.ensemble import Bootstrapper
from core.unsupervised.clustering import KMeans
from core.visualizer import draw_sample

PATH = os.path.dirname(os.path.realpath(__file__))
ratings = pd.read_csv(PATH + '/movies_csv/ratings.txt', header=None)
ratings_test = pd.read_csv(PATH + '/movies_csv/ratings_test.txt', header=None)
# movies = pd.read_csv(PATH + '/movies_csv/movies.txt', header=None)


def generate_sample(draw=False):
    w = pd.Series([.2, .5, .3])
    n = 500
    bootstrapper = Bootstrapper()
    dists = bootstrapper.sample_discrete(w, n)
    dists = pd.Series(dists)

    cov = [[1,0],[0,1]]
    mu0, n0 = [0,0], dists[dists == 0].count()
    mu1, n1 = [3,0], dists[dists == 1].count()
    mu2, n2 = [0,3], dists[dists == 2].count()
    sample = []
    sample.extend(np.random.multivariate_normal(mean=mu0, cov=cov, size=n0))
    sample.extend(np.random.multivariate_normal(mean=mu1, cov=cov, size=n1))
    sample.extend(np.random.multivariate_normal(mean=mu2, cov=cov, size=n2))
    sample = pd.DataFrame(sample)

    if draw:
        draw_sample(sample)

    return sample

def problem1():
    sample = generate_sample()
    kms = {}

    for K in [2,3,4,5]:
        km = KMeans(sample, K=K)
        km.iterate(20)
        kms[K] = km

    return kms

if __name__ == '__main__':
    problem1()