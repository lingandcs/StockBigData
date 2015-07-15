'''
Created on Apr 26, 2015

@author: lingandcs
'''
import matplotlib.pyplot as plt
import uuid
import hashlib
import logging
import gensim
import bz2
from gensim import utils, corpora, matutils, models
from gensim.models import word2vec
import numpy
import sys

if __name__ == '__main__':
     
    print id(numpy.dot) == id(numpy.core.multiarray.dot) 
    print id(numpy.dot)
    print id(numpy.core.multiarray.dot) 
    sys.exit()
    corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
           [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
           [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
           [(0, 1.0), (4, 2.0), (7, 1.0)],
           [(3, 1.0), (5, 1.0), (6, 1.0)],
           [(9, 1.0)],
           [(9, 1.0), (10, 1.0)],
           [(9, 1.0), (10, 1.0), (11, 1.0)],
           [(8, 1.0), (10, 1.0), (11, 1.0)]]
    
    tfidf = models.TfidfModel(corpus)
    vec = [(0, 1), (12, 1)]
    print tfidf[vec]
    
    texts = [['human', 'interface', 'computer'],
 ['survey', 'user', 'computer', 'system', 'response', 'time'],
 ['eps', 'user', 'interface', 'system'],
 ['system', 'human', 'system', 'eps'],
 ['user', 'response', 'time'],
 ['trees'],
 ['graph', 'trees'],
 ['graph', 'minors', 'trees'],
 ['graph', 'minors', 'survey']]
    sentences = models.word2vec.Text8Corpus('C:\\Users\\lingandcs\\Downloads\\text8')
    model = word2vec.Word2Vec(sentences, size=200)
    print model.most_similar(positive=['import', 'vector'], negative=['train'], topn=1)
    