### Daniel Kronovet
### dbk2123@columbia.edu

'''
This is the classifiers module of the library.

All functions currently operate on a pandas DataFrame or Series.
'''

import math

import numpy as np
import pandas as pd


######################################
### K-Nearest Neighbors operations ###
######################################
def get_euclidean_distance(u, v):
    '''u, v are vectors of equal length'''
    # return math.sqrt(sum([(ui - vi) ** 2 for ui, vi in zip(u, v)]))
    return math.sqrt(((u - v) ** 2).sum())

def get_ln_norm(u, n):
    # return math.sqrt(sum([ui ** n for ui in u]))
    return math.sqrt((u ** n).sum())

def get_euclidean_norm(u):
    return get_ln_norm(u, 2)

def knn_classifier(X_train, label_train, x, k=3):
    centered_X = X_train - x
    distances = centered_X.T.apply(get_euclidean_norm)
    distances.sort()
    knn = distances.iloc[:k].index
    labels = label_train.ix[knn]
    label = 0 # 0 is the name of the sole column in the labels df
    counts = labels.groupby(label).count()
    counts.sort()
    return counts[label].ix[0]

########################
### Bayes Classifier ###
########################
def get_MLE_cov(sample):
    mu = sample.mean()
    n = len(sample.columns)
    cov = pd.DataFrame(np.zeros(n**2).reshape(n,n))
    for x in sample.index:
        error = pd.DataFrame(sample.ix[x] - mu)
        cov += (error.dot(error.T)) / n
    return cov





















