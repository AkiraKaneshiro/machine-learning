### Daniel Kronovet
### dbk2123@columbia.edu

'''
This is the validations module of the library.

All functions currently operate on a pandas DataFrame or Series.
'''

import math

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