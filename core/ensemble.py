### Daniel Kronovet
### dbk2123@columbia.edu

'''
This is the ensemble methods module of the library.

All functions currently operate on a pandas DataFrame or Series.
'''

import math
import random
random.seed('Gilgamesh')

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import visualizer


class Bootstrapper(object):
    def sample_discrete(self, w, n):
        '''Sample discrete RVs with probability distribution w'''
        assert abs(w.sum() - 1) < 0.000001, 'Weights do not sum to 1!'
        w = w.cumsum()
        sample = [w[(w > random.random())].index[0] for _ in range(n)]
        return sample

    def bootstrap_sample(self, X, w, n=None, normalize_index=True):
        '''Create a boostrap sample using the weight vector w'''
        n = n or len(X)
        B_index = self.sample_discrete(w, n)
        B = X.ix[B_index]
        if normalize_index:
            self.normalize_index(B)
        return B

    def normalize_index(self, B):
        B.index = range(len(B))

class AdaBoost(Bootstrapper):
    def __init__(self, X_train, y_train, X_test, y_test, classifier=None):
        self.X_train = X_train
        self.y_train = y_train[0] if isinstance(y_train, pd.DataFrame) else y_train
        self.X_test = X_test
        self.y_test = y_test[0] if isinstance(y_test, pd.DataFrame) else y_test
        self.w = pd.Series(np.ones(len(X_train))) / len(X_train)
        self.classifier = classifier
        self.classifiers = []
        self.epsilons = []
        self.alphas = []
        self.ws = []
        self.test_preds = []
        self.train_preds = []

    def __repr__(self):
        return u'<AdaBoost for {} [t = {}]>'.format(self.classifier, self.t)

    @property
    def t(self):
        return len(self.classifiers)
        
    def iterate(self, n=1):
        for i in range(n):
            print 'Iteration', i
            self._iterate()

    def _iterate(self):
        '''Bootstrap, train, predict, update.'''
        # 1. Sample bootstrap set of size B_t using distribution w
        B_t = self.bootstrap_sample(self.X_train, self.w, normalize_index=False)
        y_t = self.y_train[B_t.index]
        self.normalize_index(B_t)
        self.normalize_index(y_t)

        # 2. Train classifier f_t using B_t
        c = self.classifier(B_t, y_t)
        self.classifiers.append(c)

        # 3. Check classifier against original data, set epsilon and alpha
        train_pred = c.classify_all(self.X_train)
        test_pred = c.classify_all(self.X_test) # For calculating test error
        epsilon = self.get_epsilon(train_pred, self.y_train)
        self.epsilons.append(epsilon)
        alpha = self.get_alpha(epsilon)
        self.alphas.append(alpha)

        # 4. Update w
        # 5. Normalize w
        self.update_w(train_pred, self.y_train, alpha)
        
        # Save train and test predictions for step t
        self.train_preds.append(alpha * train_pred)
        self.test_preds.append(alpha * test_pred)

    def get_epsilon(self, pred, y):
        misclassified = y[y != pred]
        return self.w[misclassified.index].sum()

    def get_alpha(self, epsilon):
        return 0.5 * np.log((1 - epsilon) / epsilon)

    def update_w(self, pred, y, alpha):
        self.w = self.w * (math.e ** (-alpha * (y * pred)))
        self.w = self.w / self.w.sum()
        assert abs(self.w.sum() - 1) < 0.000001, 'Weights do not sum to 1!'
        self.ws.append(self.w.copy())

    def classify(self, x):
        pred = sum([a*c.classify(x) for a,c in zip(self.alphas,self.classifiers)])
        return 1 if pred >= 0 else -1        

    def classify_all(self, X=None):
        X = X if X is not None else self.X_test
        pred = sum([a*c.classify_all(X) for a,c in zip(self.alphas,self.classifiers)])
        return self.assign_label(pred)

    def assign_label(self, pred):
        pred[pred >= 0] = 1
        pred[pred < 0] = -1
        return pred

    def get_error(self, pred, y):
        misclassified = y[y != pred]
        return len(misclassified) / float(len(y))

    def draw_params(self):
        series = {'alpha': self.alphas, 'epsilon': self.epsilons}
        visualizer.compare_series(series)

    def draw_ws(self, w_indices):
        series = {i: [w.ix[i] for w in self.ws] for i in w_indices}
        visualizer.compare_series(series)

    def train_error(self):
        pred_by_t = [sum(self.train_preds[:i+1]) for i in range(self.t)]
        pred_by_t = [self.assign_label(pred) for pred in pred_by_t]
        error_rate = [self.get_error(pred, self.y_train) for pred in pred_by_t]
        return error_rate

    def test_error(self):
        pred_by_t = [sum(self.test_preds[:i+1]) for i in range(self.t)]
        pred_by_t = [self.assign_label(pred) for pred in pred_by_t]
        error_rate = [self.get_error(pred, self.y_test) for pred in pred_by_t]
        return error_rate

    def draw_accuracy(self):
        series = {'train_err': self.train_error(), 'test_err': self.test_error()}
        visualizer.compare_series(series)








