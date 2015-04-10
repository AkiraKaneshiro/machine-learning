### Daniel Kronovet
### dbk2123@columbia.edu

'''
This is the recommendations methods module of the library.

All functions currently operate on a pandas DataFrame or Series.
'''

import math
import random
random.seed('Ozymandias')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import visualizer


class Recommender(object):
    def __init__(self, M, d=20, var=0.25, lmbda=10):
        self.d = d
        self.var = var
        self.lmbda = lmbda
        self.M = M
        self.U = self.generate_location(idx=self.M.index)
        self.V = self.generate_location(idx=self.M.columns)
        self.L = []

    @property
    def dimensions(self):
        return self.X.iloc[0].index

    def generate_location(self, idx):
        d, l, n = self.d, self.lmbda, len(idx)
        return pd.DataFrame(
                    np.random.multivariate_normal(
                        mean=np.zeros(d), cov=(1./l)*np.identity(d), size=n),
                    index = idx)

    def iterate(self, n=1):
        for i in range(n):
            print 'Updating U, iteration', i
            self.update_U()
            print 'Updating V, iteration', i
            self.update_V()

    def update_U(self):
        for i in self.M.index:
            self.update_u(i)

    def update_u(self, i):
        Mij = self.M.ix[i].dropna()
        omega_vj = Mij.index
        vj = self.V.ix[omega_vj]

        term1 = pd.DataFrame(self.lmbda * self.var * np.identity(self.d))
        term2 = vj.T.dot(vj)
        term3 = Mij.dot(vj)

        u = np.linalg.inv(term1 + term2).dot(term3)
        self.U.ix[i] = u
        
    def update_V(self):
        for j in self.M.columns:
            self.update_v(j)

    def update_v(self, j):
        Mij = self.M[j].dropna()
        omega_ui = Mij.index
        ui = self.U.ix[omega_ui]

        term1 = pd.DataFrame(self.lmbda * self.var * np.identity(self.d))
        term2 = ui.T.dot(ui)
        term3 = Mij.dot(ui)

        v = np.linalg.inv(term1 + term2).dot(term3)
        self.V.ix[j] = v




