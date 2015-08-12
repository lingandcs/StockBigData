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
import time
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
from gensim.models import Word2Vec

import logging


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
    
def create_bag_of_centroids( wordlist, word_centroid_map ):
    num_centroids = max( word_centroid_map.values() ) + 1
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids", count
    return bag_of_centroids

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
            sentsTmp = body.split('。')
            sentsList = []
            segmented = jieba.cut(preprocess(body))
            segmented = cleanText(segmented)
            sents.append(segmented)
        print '%d empty new body found' % numEmpty   
        break
        
    print '%d news loaded' % (len(sents))
    
    
    model_name = "/Users/lingandcs/Projects/FinancialNLP/data/model/200features_4minwords_14context.model"
    model = Word2Vec.load(model_name)
#     print len(model.vocab), ' words leart in model'
#     testWords = [u'习近平', u'李克强', u'钓鱼岛', u'日本', u'朝鲜', u'台湾', u'奥巴马', u'科学', u'发展观', u'科学发展观']
#     for w in testWords:
#         if w not in model.vocab:
#             print w, 'is not in vocabulary'
#             continue
#         simWords = model.most_similar(w)
#         simWords = [item[0] for item in simWords]
#         print '\n\n Most similar words to:\t', w
#         print ' '.join(simWords[:10])
#     
#     sys.exit()
    start = time.time()
    word_vectors = model.syn0
    num_clusters = word_vectors.shape[0] / 50
    kmeans_clustering = KMeans( n_clusters = num_clusters )
    idx = kmeans_clustering.fit_predict( word_vectors )
    
    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print "Time taken for K Means clustering: ", elapsed, "seconds."
    
    print model.index2word
    word_centroid_map = dict(zip( model.index2word, idx ))
    
    #build map from centroid to word list
    centroid2word = {}#key: centroid index; value: word list
    for item in word_centroid_map.items():
        w = item[0]
        idx_centroid = item[1]
        if idx_centroid not in centroid2word:
            centroid2word[idx_centroid] = []
        centroid2word[idx_centroid].append(w)
        
    for cluster in xrange(0,10):
        print "\nCluster %d" % cluster
        words = []
        for i in xrange(0,len(word_centroid_map.values())):
            if( word_centroid_map.values()[i] == cluster ):
                words.append(word_centroid_map.keys()[i])
        print ' '.join(words)

    train_centroids = np.zeros( (len(sents), num_clusters), dtype="float32" )
    counter = 0
    for s in sents:
        train_centroids[counter] = create_bag_of_centroids(s, word_centroid_map )
        counter += 1
    