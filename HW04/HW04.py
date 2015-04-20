### Daniel Kronovet (dbk2123)
### COMS W4721 HW 04
### April 14, 2015

'''
from HW04.HW04 import *
rec = problem2()

kms = problem1()
s = generate_sample()
'''

from pprint import pprint
import os
import random
random.seed('Siddhartha')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from core.supervised.ensemble import Bootstrapper
from core.unsupervised.clustering import KMeans
from core.unsupervised.factorization import Recommender
from core.visualizer import draw_sample

PATH = os.path.dirname(os.path.realpath(__file__))
ratings = pd.read_csv(PATH + '/movies_csv/ratings.txt', header=None)
ratings_test = pd.read_csv(PATH + '/movies_csv/ratings_test.txt', header=None)
with open(PATH + '/movies_csv/movies.txt') as f:
    titles = f.readlines()
titles = [m.strip() for m in titles]
# movies = pd.Series(movies)
# movies.index = (pd.Series(movies.index) + 1).values

cols = ['user', 'movie', 'rating']
ratings.columns = cols
ratings_test.columns = cols

M = pd.pivot_table(ratings, rows='user', cols='movie')['rating']
M_test = pd.pivot_table(ratings_test, rows='user', cols='movie')['rating']
M_test = M_test.drop([1325, 1414, 1577, 1604, 1637, 1681], axis=1)


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

def get_title(idx):
    return titles[idx-1]

def problem1():
    sample = generate_sample()
    kms = {}

    for K in [2,3,4,5]:
        km = KMeans(sample, K=K)
        km.iterate(20)
        kms[K] = km

    return kms

def problem2():
    rec = Recommender(M, d=20, var=0.25, lmbda=10, M_test=M_test)
    rec.iterate(2)
    return rec

def problem2_2(rec=None):
    if rec is None:
        rec = problem2()

    movies = [random.choice(rec.M.index) for _ in range(3)]
    for idx in movies:
        distances = rec.closest_items(idx, n=5)
        neighbors = {get_title(idx): distances[idx] for idx in distances.index}
        print '#####', get_title(idx)
        pprint(neighbors)

def problem2_3(rec=None):
    if rec is None:
        rec = problem2()

    U, V = rec.U, rec.V
    km = KMeans(U, K=30)
    km.iterate(10)
    centroids = km.MU[km.MU != 0].dropna()
    assert len(centroids) >= 5, 'Not enough centroids!'
    print len(centroids), 'centroids'
    ptypes = set()
    while len(ptypes) < 5:
        ptypes.add(random.choice(centroids.index))
    similarities = map_ptypes(ptypes, centroids, V)
    pprint(similarities)
    return km

def map_ptypes(ptypes, centroids, V):
    similarities = {}
    for ptype in ptypes:
        ratings = V.dot(centroids.ix[ptype])
        ratings.sort(ascending=False)
        top_10 = ratings.iloc[:10].index
        similarities[ptype] = [(get_title(i), ratings[i]) for i in top_10]
    return similarities


if __name__ == '__main__':
    problem1()
    rec = problem2()
    problem2_2(rec)
    problem2_3(rec)