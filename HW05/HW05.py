### Daniel Kronovet (dbk2123)
### COMS W4721 HW 05
### April 28, 2015

'''
from HW05.HW05 import *
nmf = problem2_2()

nmf = problem2_1()
mm = problem1()
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

from core import visualizer
from core.unsupervised.markov import MarkovModel
from core.unsupervised.factorization import NMF


PATH = os.path.dirname(os.path.realpath(__file__))

### Scores
scores = pd.read_csv(PATH + '/hw5text/cfb2014scores.csv', header=None)
scores.columns = ['E1', 'E1v', 'E2', 'E2v']
with open(PATH + '/hw5text/legend.txt') as f:
    teams = f.readlines()
def get_team(idx):
    return teams[idx-1].strip()

### Faces
faces = pd.read_csv(PATH + '/hw5text/faces.csv', header=None)

### Documents
NUM_WORDS = 3012
WORDS_IDX = range(NUM_WORDS+1)[1:]
with open(PATH + '/hw5text/nyt_data.txt') as f:
    raw_docs = f.readlines()
raw_docs = [raw_doc.strip().split(',') for raw_doc in raw_docs]

def raw_doc_to_df(raw_doc):
    doc = pd.DataFrame([word_count.split(':') for word_count in raw_doc])
    doc = doc.convert_objects(convert_numeric=True)
    doc.columns = ['word', 0]
    doc = doc.set_index('word')
    return doc

### HW Problems
def problem1():
    steps = [10, 100, 200, 1000]
    steps = [10, 90, 100, 800]
    mm = MarkovModel(scores)
    for i, step in enumerate(steps):
        print '#' * 30
        print 'Top teams for iteration', step
        mm.iterate(step)
        top_20 = mm.get_top(20)
        for team in top_20.index:
            print get_team(team), top_20[team], '\n'
    return mm

def problem2_1():
    nmf = NMF(faces, d=25, objective='sqerr')
    nmf.iterate(200)
    return nmf

def problem2_2():
    docs = pd.DataFrame(index=WORDS_IDX)
    for i, raw_doc in enumerate(raw_docs):
        if i % 100 == 0: print 'Processing document', i
        doc = raw_doc_to_df(raw_doc)
        if len(doc.index) > len(doc.index.unique()):
            doc = doc.groupby(doc.index).sum()
        doc.reindex(docs.index)
        docs[i] = doc
    docs = docs.fillna(0)

    nmf = NMF(docs, d=25, objective='dvrg')
    # nmf = NMF(faces, d=25, objective='dvrg')
    # nmf.iterate(200)
    return nmf

if __name__ == '__main__':
    problem1()