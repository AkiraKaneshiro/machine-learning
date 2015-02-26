### Daniel Kronovet
### dbk2123@columbia.edu

'''
This is the regressions module of the library.

All functions currently operate on a pandas DataFrame or Series.
'''

import math

import numpy as np


################################
### Least Squares operations ###
################################
def get_w_hat(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

def get_y_hat(w_hat, X):
    return X.dot(w_hat)

#####################################
### Maximum Likelihood Estimators ###
#####################################
def MLE_mean(sample):
    '''Get MLE mean from a pd.Series'''
    data = sample.values
    mu = sum(data) / float(len(data)) # Unbiased estimator
    assert mu - data.mean() < .00001 # Error tolerance
    return mu

def MLE_var(sample):
    '''Get MLE variance from a pd.Series'''
    mu = MLE_mean(sample)
    data = sample.values
    var = sum([(x - mu) ** 2 for x in data]) / float(len(data)-1) # Unbiased estimator
    assert var - (sample.std() ** 2) < .00001 # Error tolerance
    return var