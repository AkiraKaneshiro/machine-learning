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
    def __init__(self, M, d=20, var=0.25, lmbda=10, M_test=None):
        self.d = d
        self.var = var
        self.lmbda = lmbda
        self.M = M
        self.M_test = M_test
        self.U = self.generate_location(idx=self.M.index)
        self.V = self.generate_location(idx=self.M.columns)
        self.RMSE = []
        self.LJL = []
        self.steps = 0

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
        for _ in range(n):
            print 'Iteration', len(self.LJL)+1 
            self.update_U()
            self.update_V()
            self.RMSE.append(self.get_RMSE())
            self.LJL.append(self.get_LJL())

    def update_U(self):
        for i in self.M.index:
            self.update_u(i)

    def update_u(self, i):
        self.steps += 1
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
        self.steps += 1
        Mij = self.M[j].dropna()
        omega_ui = Mij.index
        ui = self.U.ix[omega_ui]

        term1 = pd.DataFrame(self.lmbda * self.var * np.identity(self.d))
        term2 = ui.T.dot(ui)
        term3 = Mij.dot(ui)

        v = np.linalg.inv(term1 + term2).dot(term3)
        self.V.ix[j] = v

    def recommend(self):
        return self.U.dot(self.V.T)

    def get_RMSE(self):
        '''Get Root Mean Square Error'''
        if self.M_test is None:
            raise Exception('Must set M_test attribute to test set')
        errors = []
        M_rec = self.recommend()
        for i in self.M_test.index:
            for j in self.M_test.ix[i].dropna().index:
                try:
                    error = M_rec.ix[i][j] - self.M_test.ix[i][j]
                    errors.append(error)
                except KeyError as ex:
                    # print ex
                    pass
        return math.sqrt(sum([e**2 for e in errors]) / len(errors))

    def get_LJL(self):
        '''Get the log joint likelihood'''
        lnpUi = -(self.lmbda / 2.) * (self.U ** 2).sum(axis=1).sum()
        lnpVj = -(self.lmbda / 2.) * (self.V ** 2).sum(axis=1).sum()
        lnpMij = 0
        for i in self.M.index:
            Mi = self.M.ix[i].dropna()
            omega_vj = Mi.index
            ui = self.U.ix[i]
            Vj = self.V.ix[omega_vj]
            lnpMij += ((Mi - ui.dot(Vj.T)) ** 2).sum()
        lnpMij *= -(1./(2*self.var))
        return lnpMij + lnpUi + lnpVj






