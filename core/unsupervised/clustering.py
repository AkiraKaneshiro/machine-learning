### Daniel Kronovet
### dbk2123@columbia.edu

'''
This is the clustering methods module of the library.

All functions currently operate on a pandas DataFrame or Series.
'''

import math
import random
random.seed('Merkaba')

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import visualizer


class KMeans(object):
    def __init__(self, X, k=3):
        self.X = X
        self.k = k
        self.c = pd.Series(0, index=X.index)
        self.MU = pd.DataFrame(0, index=range(k), columns=self.dimensions)

    @property
    def dimensions(self):
        return self.X.iloc[0].index

    def update_MU(self):
        '''
        For each mu_k, get all data points where c == k,
            update mu_k with mean for that sample.
        '''
        for i in self.MU.index:
            # class_index = self.c[c == i].index
            class_index = self.c == i
            x_class = X.ix[class_index]
            self.MU.ix[i] = self.sample_mu(x_class)

    def sample_mu(self, sample):
        return sample.mean() / len(sample)

    def update_c(self):
        '''
        dataframe of N x K
        Each value is the distance of x_i to mu_k
        For each x_i, assign c as the min
        '''
        distances = pd.DataFrame(0, index=self.X.index, columns=self.MU.index)
        for i in self.MU.index:
            distances[i] = self.X - self.MU.ix[i]

        for i in self.X.index:
            self.c[i] = distances.ix[i].min() ## Not actually min, but the column name of the min col



