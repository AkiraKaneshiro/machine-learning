### Daniel Kronovet
### dbk2123@columbia.edu

'''
This is the learning module of the library.

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

######################
### Error Checking ###
######################
def get_MAE(y, y_hat):
    '''Get Mean Absolute Error from a pd.Series'''
    return (y - y_hat).abs().sum() / len(y)

def get_RMSE(y, y_hat):
    '''Get Root Mean Square Error from a pd.Series'''
    return math.sqrt(((y - y_hat) ** 2).sum() / len(y))

def log_likelihood(sample):
    n = len(sample)
    mu = MLE_mean(sample)
    var = MLE_var(sample)
    term1 = -(n/2.) * math.log(2 * math.pi)
    term2 = -(n/2.) * math.log(var)
    term3 = -(1 / (2. * var)) * sum([(x - mu) ** 2 for x in sample])
    return term1 + term2 + term3