### Daniel Kronovet (dbk2123)
### COMS W4721 HW 02
### March 3, 2015

import os

import pandas as pd

from core import classifiers

path = os.path.dirname(os.path.realpath(__file__))

# random.seed('codex seraphinianus')

X_test = pd.read_csv(path + '/mnist_csv/Xtest.txt', header=None)
label_test = pd.read_csv(path + '/mnist_csv/label_test.txt', header=None)
X_train = pd.read_csv(path + '/mnist_csv/Xtrain.txt', header=None)
label_train = pd.read_csv(path + '/mnist_csv/label_train.txt', header=None)
Q = pd.read_csv(path + '/mnist_csv/Q.txt', header=None)

if __name__ == '__main__':
    pass
