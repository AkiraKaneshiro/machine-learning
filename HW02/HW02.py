### Daniel Kronovet (dbk2123)
### COMS W4721 HW 02
### March 3, 2015

'''
from HW02 import *
import core.classifiers as c
logit = c.Logit(X_train, label_train, X_test, label_test, Q)
bayes = c.Bayes(X_train, label_train, X_test, label_test, Q)
knn = c.KNN(X_train, label_train, X_test, label_test, Q)
'''

import os

import pandas as pd

from core import classifiers

path = os.path.dirname(os.path.realpath(__file__))

X_test = pd.read_csv(path + '/mnist_csv/Xtest.txt', header=None)
label_test = pd.read_csv(path + '/mnist_csv/label_test.txt', header=None)
X_train = pd.read_csv(path + '/mnist_csv/Xtrain.txt', header=None)
label_train = pd.read_csv(path + '/mnist_csv/label_train.txt', header=None)
Q = pd.read_csv(path + '/mnist_csv/Q.txt', header=None)

def problem_3a():
    knn = classifiers.KNN(X_train, label_train, X_test, label_test)
    for k in [1,2,3,4,5]:
        knn.k = k
        knn.run_all()
        print 'Confusion matrix for k=', k
        print knn.confusion_matrix
        print 'Prediction accuracy for k=', k
        print knn.prediction_accuracy()
        print
    return knn

def problem_3b():
    bayes = classifiers.MultiClassBayes(X_train, label_train, X_test, label_test)
    bayes.run_all()
    print 'Confusion matrix for bayes:'
    print bayes.confusion_matrix
    print 'Prediction accuracy for bayes:'
    print bayes.prediction_accuracy()
    print
    return bayes

def problem_3c():
    logit = classifiers.Logit(X_train, label_train, X_test, label_test, Q)
    logit.iterative_update(1000)
    logit.run_all()
    print 'Confusion matrix for logistic regression:'
    print logit.confusion_matrix
    print 'Prediction accuracy for logistic regression:'
    print logit.prediction_accuracy()
    print
    return logit

if __name__ == '__main__':
    problem_3a()
    problem_3b()
    problem_3c()


