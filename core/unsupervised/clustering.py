### Daniel Kronovet
### dbk2123@columbia.edu

'''
This is the clustering methods module of the library.

All functions currently operate on a pandas DataFrame or Series.
'''

import math
import random
random.seed('Merkaba')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import visualizer


class KMeans(object):
    def __init__(self, X, K=3):
        self.X = X
        self.K = K
        self.MU = self.generate_MU()
        self.c = pd.Series(0, index=X.index)
        self.L = []

    @property
    def dimensions(self):
        return self.X.iloc[0].index

    def generate_MU(self):
        X, K, dim = self.X, self.K, self.dimensions
        start_points = set()
        while len(start_points) < K:
            start_points.add(random.choice(X.index))
        return pd.DataFrame(X.ix[start_points], index=range(K), columns=dim)

    def iterate(self, n=1):
        for _ in range(n):
            self.update_c()
            self.update_MU()
            self.update_L()

    def update_MU(self):
        '''
        For each mu_k, get all data points where c == k,
            update mu_k with mean for that sample.
        '''
        for k in self.MU.index:
            x_class = self.get_class_data(k)
            self.MU.ix[k] = self.sample_mu(x_class)

    def get_class_data(self, k):
            class_index = self.c == k
            return self.X.ix[class_index]

    def sample_mu(self, sample):
        return (sample.sum() / len(sample)) if len(sample) else 0

    def update_c(self):
        '''
        dataframe of N x K
        Each value is the distance of x_i to mu_k
        For each x_i, assign c as the min
        '''
        distances = pd.DataFrame(0, index=self.X.index, columns=self.MU.index)
        for i in self.MU.index:
            centered_X = self.X - self.MU.ix[i]
            distances[i] = (centered_X ** 2).sum(axis=1)

        for i in self.X.index:
            self.c[i] = distances.ix[i].idxmin()

    def update_L(self):
        l = 0
        for k in self.MU.index:
            x_class = self.get_class_data(k)
            l += ((x_class - self.MU.ix[k]) ** 2).sum(axis=1).sum()
        self.L.append(l)

    def draw_sample(self):
        plt.scatter(self.X[0], self.X[1])
        plt.scatter(self.MU[0], self.MU[1], color='r', marker='*')



