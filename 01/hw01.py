import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

random.seed('go bears!')
TEST_SIZE = 20

X_orig = pd.read_csv('data_csv/X.txt', header=None)
y_orig = pd.read_csv('data_csv/y.txt', header=None)
columns = ['intrcpt', 'num cyl', 'displacement', 'hp', 'weight', 'accel', 'year']
X_orig.columns = columns
X, y = X_orig, y_orig

### Data processing utilities
def _shuffle_data(X, y):
    i = random.randint(0, len(X)) - 1 # -1 to avoid 'out of range' error
    assert X.index[i] == y.index[i], 'Data not aligned'
    rows = list(X.index)
    random.shuffle(rows)
    X = X.reindex(rows)
    y = y.reindex(rows)
    X.index = range(len(X))
    y.index = range(len(y))
    return X, y

def _split_data(X, y, split_size=TEST_SIZE):
    # Testing data (smaller)
    A = X.iloc[:split_size]
    a = y.iloc[:split_size]

    # Training data (larger)
    B = X.iloc[split_size:]
    b = y.iloc[split_size:]
    return A, a, B, b

def prepare_data(X, y):
    X, y = _shuffle_data(X, y)
    return _split_data(X, y)

def generate_p(X, p, has_intercept=True):
    X_p = X.copy()
    if p <= 1:
        return X_p

    for degree in range(p+1)[2:]:
        X_deg = X.copy() ** degree
        if has_intercept:
            intercept_col = X_deg.columns[0]
            del X_deg[intercept_col]
        X_deg.columns = ['{}^{}'.format(col, degree) for col in X_deg.columns]
        X_p = pd.concat([X_p, X_deg], axis=1)

    # Test for correctness
    test_vector = random.choice(X_p.T)
    test_var = random.choice(columns[1:])
    test_var_p = '{}^{}'.format(test_var, p)
    assert test_vector.ix[test_var] ** p == test_vector.ix[test_var_p]

    return X_p

### Least Squares operations
def get_w_hat(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

def get_y_hat(w_hat, X):
    return X.dot(w_hat)

def get_MAE(y, y_hat):
    return (y - y_hat).abs().sum() / len(y)

def get_RMSE(y, y_hat):
    return math.sqrt(((y - y_hat) ** 2).sum() / len(y))

### Log likelihood operations
def log_likelihood(sample):
    n = len(sample)
    mu = MLE_mu(sample)
    var = MLE_var(sample)
    term1 = -(n/2.) * math.log(2 * math.pi)
    term2 = -(n/2.) * math.log(var)
    term3 = -(1 / (2. * var)) * sum([(x - mu) ** 2 for x in sample])
    return term1 + term2 + term3

def MLE_mu(sample):
    data = sample.values
    mu = sum(data) / float(len(data))
    assert mu - data.mean() < .00001 # Error tolerance
    return mu

def MLE_var(sample):
    mu = MLE_mu(sample)
    data = sample.values
    var = sum([(x - mu) ** 2 for x in data]) / float(len(data))
    assert var - (data.std() ** 2) < .001 # Error tolerance
    return var

### Visualization methods
def plot_errors_hist(errors):
    colors = ['green', 'blue', 'red', 'orange']
    fig, axes = plt.subplots(int(math.ceil(len(errors) / 2.0)), 2)
    for i, p in enumerate(errors):
        plot = axes[i / 2][i % 2]
        data = errors[p].values
        plot.hist(data, bins=50, label='p={}'.format(p), color=colors[i])
        plot.legend(loc='best')
    plt.show()

def print_RMSE_by_p(RMSEs):
    print 40 * '#'
    print 'RMSE as a function of p:'
    print 'p', '\tMean', '\t\tStd'
    for p in RMSEs:
        data = RMSEs[p]
        print p, '\t', data.mean(), '\t', data.std()
    print

def print_log_likelihood_by_p(errors):
    print 50 * '#'
    print 'Error distribution and likelihood as a function of p:'
    print 'p', '\tMean', '\t\t\tVar', '\t\tLog Likelihood'
    for p in errors:
        data = errors[p]
        print p, '\t', data.mean(), '\t', data.std(), '\t', log_likelihood(data)
    print


########################
### Analysis scripts ###
########################
def part1():
    X, y = X_orig.copy(), y_orig.copy()
    X_test, y_test, X_train, y_train = prepare_data(X, y)
    w_hat = get_w_hat(X_train, y_train)
    print 'Sample w hat:'
    print w_hat
    print 

    y_hat = get_y_hat(w_hat, X_test)
    assert len(y_test) == len(y_hat)

    MAE = get_MAE(y_test, y_hat)
    print 'Sample MAE:', MAE[0]
    print

    MAEs = []
    while len(MAEs) < 1000:
        if not len(MAEs) % 100: print 'Finished {} loops'.format(len(MAEs)) 

        try:
            X_test, y_test, X_train, y_train = prepare_data(X, y)
            w_hat = get_w_hat(X_train, y_train)
        except:
            continue

        y_hat = get_y_hat(w_hat, X_test)
        MAE = get_MAE(y_test, y_hat)
        MAEs.append(MAE)

    MAEs = pd.Series(MAEs)
    print 'MAE mean:', MAEs.mean()
    print 'MAE standard deviation:', MAEs.std()
    print
    return MAEs

def part2(p_list=None):
    p_list = p_list or [1,2,3,4]
    RMSE_by_p = {}
    errors_by_p = {}
    for p in p_list:
        print 'For p = ', p
        X, y = X_orig.copy(), y_orig.copy()
        X = generate_p(X, p)

        RMSEs = []
        errors = pd.DataFrame()
        while len(RMSEs) < 1000:
            if not len(RMSEs) % 100: print 'Finished {} loops'.format(len(RMSEs)) 
            try:
                X_test, y_test, X_train, y_train = prepare_data(X, y)
                w_hat = get_w_hat(X_train, y_train)
            except:
                continue
            y_hat = get_y_hat(w_hat, X_test)
            errors = errors.append(y_test - y_hat)
            RMSE = get_RMSE(y_test, y_hat)
            RMSEs.append(RMSE)
            if len(RMSEs) == 1:
                print 'w hat:'
                print w_hat
  
        errors.index = range(len(errors))
        errors = errors[0] # Convert to pd.Series
        RMSEs = pd.Series(RMSEs)
        print 'RMSE mean:', RMSEs.mean()
        print 'RMSE standard deviation:', RMSEs.std()
        print

        RMSE_by_p[p] = RMSEs
        errors_by_p[p] = errors

    return RMSE_by_p, errors_by_p

if __name__ == '__main__':
    MAES = part1()
    RMSE, errors = part2()
    print_RMSE_by_p(RMSE) # 3.2.a
    print_log_likelihood_by_p(errors) # 3.2.c
    plot_errors_hist(errors) # 3.2.b







