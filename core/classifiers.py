### Daniel Kronovet
### dbk2123@columbia.edu

'''
This is the classifiers module of the library.

All functions currently operate on a pandas DataFrame or Series.
'''

import math

import numpy as np


######################################
### K-Nearest Neighbors operations ###
######################################
def get_euclidean_distance(u, v):
    '''u, v are vectors of equal length'''
    return math.sqrt(sum([(ui - vi) ** 2 for ui, vi in zip(u, v)]))

def get_euclidean_norm(u):
    return math.sqrt(sum([ui ** 2 for ui in u]))

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

def knn_classifier_short(X_train, label_train, x, k=3):
    distances = (X_train - x).T.apply(get_euclidean_norm)
    distances.sort()
    counts = label_train.ix[distances.iloc[:k].index].groupby(0).count()
    counts.sort()
    return counts[0].ix[0]