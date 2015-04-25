### Daniel Kronovet
### dbk2123@columbia.edu

'''
This is the Markov-related methods module of the library.

All functions currently operate on a pandas DataFrame or Series.
'''

import math
import random
random.seed('Arizal')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import visualizer

class MarkovModel(object):
    def __init__(self, results):
        self.build_M_ranking(results)
        self.w = pd.Series(1./len(self.M), index=self.M.index)
        self.nrm_u1 = self.get_normalized_un(n=1)
        self.l1_diffs = []

    def build_M_ranking(self, results):
        '''Build transition matrix M for competing teams using game results'''
        states = []
        for idx in ['T1', 'T2']:
            states.extend(results[idx])
        states = set(states)
        self.M = pd.DataFrame(0, index=states, columns=states)
        for i in results.index:
            game = results.ix[i]
            self.update(game['T1'], game['T2'], game['T1p'], game['T2p'])

        self.M = self.M.divide(self.M.sum(axis=1), axis=0)
        assert abs(self.M.sum(axis=1).max() - 1) < 0.0001 # Normalized correctly
        assert abs(self.M.sum(axis=1).min() - 1) < 0.0001 # Normalized correctly

    def update(self, t1_idx, t2_idx, t1_score, t2_score):
        t1_win = 1 if t1_score > t2_score else 0
        t2_win = 1 if t2_score > t1_score else 0
        t1_edge = float(t1_score) / (t1_score + t2_score)
        t2_edge = float(t2_score) / (t1_score + t2_score)
        self.M.ix[t1_idx, t1_idx] += (t1_win + t1_edge) 
        self.M.ix[t2_idx, t1_idx] += (t1_win + t1_edge) 
        self.M.ix[t2_idx, t2_idx] += (t2_win + t2_edge) 
        self.M.ix[t1_idx, t2_idx] += (t2_win + t2_edge)

    def iterate(self, n=1):
        for _ in range(n):
            self.w = self.w.dot(self.M)
            self.update_l1_diff()

    def update_l1_diff(self):
        diff = np.abs(self.w - self.nrm_u1).sum()
        self.l1_diffs.append(diff)

    def get_top(self, n):
        w = self.w.copy()
        w.sort(ascending=False)
        return w[:n]

    def get_normalized_un(self, n=1):
        '''nth eigenvector of M'''
        lmbda1, un = self.get_eigenvector(n)
        un.index = self.M.index
        return un / un.sum()

    def get_eigenvectors(self):
        vals, vecs = np.linalg.eig(self.M.T)
        # Vec have l2 norm of 1: (vecs**2).sum()
        return pd.Series(vals), pd.DataFrame(vecs)

    def get_eigenvector(self, n=1):
        assert n >= 1, 'Eigenvector {} does not exist'.format(n)
        vals, vecs = self.get_eigenvectors()
        vals.sort(ascending=False)
        idx = vals.index[n-1]
        val, vec = vals.ix[idx], vecs[idx]
        print 'Index', idx, (vec**2).sum(), val
        assert abs((vec**2).sum() - 1) < 0.0001 # l2 norm == 1
        return val, vec












