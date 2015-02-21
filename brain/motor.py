### Daniel Kronovet
### dbk2123@columbia.edu

'''
This is the data wrangling module of the library.

All functions currently operate on a pandas DataFrame.
'''

import random


def shuffle_and_split(X, y, split_size=20):
    X, y = shuffle_data(X, y)
    return split_data(X, y)

def shuffle_data(X, y, verify=True):
    if verify:
        i = random.randint(0, len(X)-1) # -1 to avoid 'out of range' error
        assert X.index[i] == y.index[i], 'Data not aligned'

    rows = list(X.index)
    random.shuffle(rows)
    X = X.reindex(rows)
    y = y.reindex(rows)
    X.index = range(len(X))
    y.index = range(len(y))
    return X, y

def split_data(X, y, split_size):
    # Testing data (smaller)
    A = X.iloc[:split_size]
    a = y.iloc[:split_size]

    # Training data (larger)
    B = X.iloc[split_size:]
    b = y.iloc[split_size:]
    return A, a, B, b

def generate_p(X, p, has_intercept=True, verify=True):
    if p <= 1:
        return X_p

    if has_intercept:
        intercept = X_p[X_p.columns[0]]
        del X_p[X_p.columns[0]]

    for degree in range(p+1)[2:]: # [1, ... ,p]
        X_deg = X.copy() ** degree
        X_deg.columns = ['{}^{}'.format(col, degree) for col in X_deg.columns]
        X_p = pd.concat([X_p, X_deg], axis=1)

    if has_intercept:
        X_p = pd.concat([intercept, X_p], axis=1)

    if verify:
        test_vector = random.choice(X_p.T)
        test_var = random.choice(columns[1:])
        test_var_p = '{}^{}'.format(test_var, p)
        assert test_vector.ix[test_var] ** p == test_vector.ix[test_var_p]

    return X_p