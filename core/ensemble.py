### Daniel Kronovet
### dbk2123@columbia.edu

'''
This is the ensemble methods module of the library.

All functions currently operate on a pandas DataFrame or Series.
'''

# import math
import random
random.seed('Gilgamesh')

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd


class Bootstrapper(object):

    def sample_discrete(self, w, n):
        '''Sample discrete RVs with probability distribution w'''
        assert abs(w.sum() - 1) < 0.000001, 'Weights do not sum to 1!'
        w = w.cumsum()
        sample = [w[(w > random.random())].index[0] for _ in range(n)]
        return sample

    def boostramp_sample(self, X, w, n=None):
        '''Create a boostrap sample using the weight vector w'''
        n = n or len(X)
        B_index = self.sample_discrete(w, n)
        B = X.ix[B_index]
        return B

