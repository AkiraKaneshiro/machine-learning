### Daniel Kronovet
### dbk2123@columbia.edu

'''
This is the recommendations methods module of the library.

All functions currently operate on a pandas DataFrame or Series.
'''

import math
import random
random.seed('Polanyi')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SQERR = 'sqerr'
DVRG = 'dvrg'
TINY = 10 ** -16

class NMF(object):
    def __init__(self, X, d=25, objective=SQERR):
        self.X = X
        self.d = d
        self.objective = objective
        self.H = self.generate_location(self.X.columns).T
        self.W = self.generate_location(self.X.index)
        self.objs = []

    @property
    def WH(self):
        return self.W.dot(self.H)

    def generate_location(self, idx):
        return np.abs(pd.DataFrame(np.random.randn(len(idx), self.d), index=idx))

    def iterate(self, n=1):
        for _ in range(n):
            print 'Iteration', len(self.objs)
            self.update()

    def update(self):
        if self.objective == SQERR:
            self.update_sqerr()
        elif self.objective == DVRG:
            self.update_dvrg()
        else:
            raise ValueError('Unrecognized objective function!')

    # Squared Error Objective
    def update_sqerr(self):
        self.update_H_sqerr()
        self.update_W_sqerr()
        self.update_objective_sqerr()

    def update_objective_sqerr(self):
        obj = ((self.X - self.WH) ** 2).sum().sum()
        self.objs.append(obj)

    def update_H_sqerr(self):
        num = self.W.T.dot(self.X)
        denom = self.W.T.dot(self.W).dot(self.H)
        self.H = self.H.mul(num.div(denom+TINY))

    def update_W_sqerr(self):
        num = self.X.dot(self.H.T)
        denom = self.W.dot(self.H).dot(self.H.T)
        self.W = self.W.mul(num.div(denom+TINY))

    # Divergence Objective
    def update_dvrg(self):
        self.update_H_dvrg()
        self.update_W_dvrg()
        self.update_objective_dvrg()

    def update_objective_dvrg(self):
        WH = self.WH
        obj = -((self.X.mul(np.log(WH)) - WH).sum().sum())
        # obj = (self.X.mul(np.log(1./WH)) + WH).sum().sum()
        self.objs.append(obj)

    def update_H_dvrg(self):
        purple = self.X.div(self.WH+TINY)
        nrmlW = (self.W / self.W.sum()).T
        self.H = self.H.mul(nrmlW.dot(purple))

    def update_W_dvrg(self):
        purple = self.X.div(self.WH+TINY)
        nrmlH = (self.H.T / self.H.T.sum())
        self.W = self.W.mul(purple.dot(nrmlH))

    def get_w_by_h(self, h):
        idx = h.idxmax()
        return self.W[idx]

class PMF(object):
    def __init__(self, M, d=20, var=0.25, lmbda=10, M_test=None):
        self.M = M
        self.d = d
        self.var = var
        self.lmbda = lmbda
        self.M_test = M_test
        self.U = self.generate_location(idx=self.M.index)
        self.V = self.generate_location(idx=self.M.columns)
        self.RMSE = []
        self.LJL = []

    @property
    def dimensions(self):
        return self.X.iloc[0].index

    def generate_location(self, idx):
        d, l, n = self.d, self.lmbda, len(idx)
        return pd.DataFrame(
                    np.random.multivariate_normal(
                        mean=np.zeros(d), cov=(1./l)*np.identity(d), size=n),
                    index=idx)

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

    def recommend(self, as_int=True):
        rec = self.U.dot(self.V.T)
        if as_int:
            rec = np.round(rec)
            rec[rec < 0] = 0
            rec[rec > 5] = 5
        return rec

    def get_RMSE(self):
        '''Get Root Mean Square Error'''
        if self.M_test is None:
            raise Exception('Must set M_test attribute to test set')
        errors = []
        M_rec = self.recommend()
        for i in self.M_test.index:
            Mi = self.M_test.ix[i].dropna()
            omega_vj = Mi.index
            error = Mi - M_rec.ix[i, omega_vj]
            errors.extend(error.values)
        errors = pd.Series(errors)
        return math.sqrt((errors ** 2).mean())

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

    def closest_items(self, j, n=5):
        item = self.V.ix[j]
        distances = ((self.V - item) ** 2).sum(axis=1)
        distances.sort(ascending=False)
        return distances.iloc[:n]






