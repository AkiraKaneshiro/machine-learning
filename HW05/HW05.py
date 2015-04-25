### Daniel Kronovet (dbk2123)
### COMS W4721 HW 05
### April 28, 2015

'''
from HW05.HW05 import *
mm = problem1()

nmf = problem2_2()
nmf.iterate(50)
'''

from datetime import datetime
from pprint import pprint
import os
import random
random.seed('Schumacher')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.unsupervised.markov import MarkovModel
from core.unsupervised.factorization import NMF


PATH = os.path.dirname(os.path.realpath(__file__))

start = datetime.now()
### Scores
scores = pd.read_csv(PATH + '/hw5text/cfb2014scores.csv', header=None)
scores.columns = ['T1', 'T1p', 'T2', 'T2p']
with open(PATH + '/hw5text/legend.txt') as f:
    teams = f.readlines()
def get_team(idx):
    return teams[idx-1].strip()
print (datetime.now() - start).total_seconds(), 'time to load scores.'

### Faces
faces = pd.read_csv(PATH + '/hw5text/faces.csv', header=None)
print (datetime.now() - start).total_seconds(), 'time to load faces.'

### Documents
# with open(PATH + '/hw5text/nyt_data.txt') as f:
#     raw_docs = f.readlines()
# raw_docs = [raw_doc.strip().split(',') for raw_doc in raw_docs]

# def raw_doc_to_df(raw_doc):
#     doc = pd.DataFrame([word_count.split(':') for word_count in raw_doc])
#     doc.columns = ['word', 0]
#     doc = doc.set_index('word')
#     return doc

# docs = raw_doc_to_df(raw_docs[0])
# for i, raw_doc in enumerate(raw_docs[1:]):
#     doc = raw_doc_to_df(raw_docs[i])
#     doc.
#     docs[i+1] = doc
# print (datetime.now() - start).total_seconds(), 'time to load docs.'

### HW Problems
def problem1():
    steps = [10, 100, 200, 1000]
    mm = MarkovModel(scores)
    for i, step in enumerate(steps):
        print '#' * 30
        print 'Top teams for iteration', step
        step_size = step - sum(steps[:i])
        mm.iterate(step_size)
        top_20 = mm.get_top(20)
        for team in top_20.index:
            print get_team(team), top_20[team], '\n'
    return mm

def problem2_1():
    nmf = NMF(faces, d=25, objective='sqerr')
    return nmf

def problem2_2():
    # nmf = NMF(docs, d=25, objective='dvrg')
    nmf = NMF(faces, d=25, objective='dvrg')
    return nmf

if __name__ == '__main__':
    problem1()