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
    def __init__(self, T):
        self.build_M_ranking(T)
        self.w = pd.Series(1./len(self.M), index=self.M.index)

    def build_M_ranking(self, T):
        '''Build transition matrix M for competing teams using transitions T'''
        states = []
        for idx in ['T1', 'T2']:
            states.extend(T[idx])
        states = set(states)
        self.M = pd.DataFrame(0, index=states, columns=states)

        for i in T.index:
            game = T.ix[i]
            if game['T1p'] > game['T2p']:
                self.update_win(game['T1'], game['T2'], game['T1p'], game['T2p'])
            else:
                self.update_win(game['T2'], game['T1'], game['T2p'], game['T1p'])

        self.M = self.M.divide(self.M.sum(axis=1), axis=0)

    def update_win(self, winner_idx, loser_idx, winner_score, loser_score):
        update_value = float(winner_score) / (winner_score + loser_score)
        self.M.ix[winner_idx, winner_idx] += update_value
        self.M.ix[loser_idx, winner_idx] += update_value

    def iterate(self, n=1):
        for _ in range(n):
            self.w = self.w.dot(self.M)

    def get_top(self, n):
        w = self.w.copy()
        w.sort(ascending=False)
        return w[:n]

    def get_eigenvector(self, n):
        vals, vecs = np.linalg.eig(self.M.T)
        vals, vecs = pd.Series(vals), pd.DataFrame(vecs)









