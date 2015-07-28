#!/usr/bin/python
# -*- coding: utf8 -*-
'''
Created on Jul 14, 2015

@author: lingandcs
'''

import jieba
import pandas as pd
import scipy
import numpy
import sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans
import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import glob

import logging
from gensim.models import word2vec


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

def cleanText(doc):
    if isinstance(doc, basestring):
        doc = doc.strip().split()
    #othewise, it's a word list
    
    newDoc = []
    for word in doc:
        if word.isdigit() == False:
            newDoc.append(word.strip())
        else:
#             print word
            pass
    
    if isinstance(doc, basestring):
        return ' '.join(newDoc)
    else:
        return newDoc

def preprocess(doc):
#     print doc
    doc = doc.replace('视频', '')
    return doc
    

if __name__ == '__main__':

    
    dataFolder = '/Users/lingandcs/Projects/FinancialNLP/data/xwlb/*.csv'
    sents = []
    
    
    for fname in glob.glob(dataFolder):
        print 'processing', fname 
        numEmpty = 0
        df = pd.read_csv(fname)
        for i, row in df.iterrows():
            if i % 1000 == 0:
                print '%d news loaded' % i
            dt = row[0]
            title = row[1]
            body = row[2]
            if body in [np.NAN, np.nan, np.NaN, None, '']:
                numEmpty += 1
#                 print dt
#                 print title
#                 print body
                continue
            segmented = jieba.cut(preprocess(body))
            segmented = cleanText(segmented)
            sents.append(segmented)
        print '%d empty new body found' % numEmpty    
        
    print '%d news loaded' % (len(sents))
    
    
    num_features = 150    # Word vector dimensionality                      
    min_word_count = 10   # Minimum word count                        
    num_workers = 2       # Number of threads to run in parallel
    contextWindow = 5          # Context window size                                                                                    
    downsampling = 1000   # Downsample setting for frequent words
    
    # Initialize and train the model (this will take some time)
    from gensim.models import word2vec
    print "Training model..."
#     model = word2vec.Word2Vec(sents, workers=num_workers, \
#                 size=num_features, min_count = min_word_count, \
#                 window = contextWindow, sample = downsampling, sg=0)
    model = word2vec.Word2Vec(sents, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                )
    # If you don't plan to train the model any further, calling 
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)
    
    # It can be helpful to create a meaningful model name and 
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "/Users/lingandcs/Projects/FinancialNLP/data/model/200features_4minwords_14context.model"
    model.save(model_name)
    print len(model.vocab), ' words leart in model'
    testWords = [u'习近平', u'李克强', u'钓鱼岛', u'日本', u'朝鲜', u'台湾', u'奥巴马', u'科学', u'发展观', u'科学发展观']
    for w in testWords:
        if w not in model.vocab:
            print w, 'is not in vocabulary'
            continue
        simWords = model.most_similar(w)
        simWords = [item[0] for item in simWords]
        print '\n\n Most similar words to:\t', w
        print ' '.join(simWords[:10])
    
    sys.exit()
    
    vectorizer = TfidfVectorizer(max_features=150)
    X = vectorizer.fit_transform(sents)
    print 'feature matrix shape\t', X.shape
    fns = vectorizer.get_feature_names()
#     for fn in fns:
#         print fn
    print '%d features got!' % len(fns)
    print X[0].shape
#     for i in xrange(X[0].shape[1]):
#         print X[0,i],fns[i]
    
    t0 = time()  
    
    k = 30
    km = KMeans(n_clusters=k, init='k-means++', max_iter=1000, n_init=1,
                verbose=False)
    t0 = time()
    print X.shape
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    print '%d centroids got' % len(order_centroids)
    terms = vectorizer.get_feature_names()
    for i in range(5):
        print "Cluster %d:" % i
#         print order_centroids[i]
        centroidWords = [fns[ind] for ind in order_centroids[i, :20]]
        print ' '.join(centroidWords)