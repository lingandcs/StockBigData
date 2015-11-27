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
from gensim.models import Phrases
import re
import csv


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
    doc = doc.replace('新闻联播', '')
    doc = doc.replace('本台消息', '')
#     doc = doc.replace('新闻联播', '')
#     doc = doc.replace('新闻联播', '')
#     doc = doc.replace('【', '')
#     doc = doc.replace('】', '')
    doc = re.sub(r'\[\]','',doc)
    doc = re.sub(r'【|】','',doc)
    doc = re.sub(r'\d+','DIGIT',doc)
    return doc
    

if __name__ == '__main__':

    
    dataFolder = '/home/lingandcs/workspace/FinancialNLP/data/*.csv'
    sents = []
    i = 0
    numNoNewsBody = 0
    
    for fname in glob.glob(dataFolder):
        
        fi = open(fname, 'rb')
        data = fi.read()
        fi.close()
        fo = open(fname, 'wb')
        fo.write(data.replace('\x00', ''))
        fo.close()
        
        print 'processing', fname 
        numEmpty = 0
#         df = pd.read_csv(fname)
        handle = open(fname, 'rU')
        csvReader = csv.reader(handle)
        for row in csvReader:
            if len(row) != 3:
                numNoNewsBody += 1
                print numNoNewsBody, ' news has no body', numNoNewsBody
#                 sys.exit()
                continue            
            i += 1
            if i % 1000 == 0:
                print '%d news loaded' % i
            dt = row[0]
            title = row[1]
            body = row[2]
            if body in [np.NAN, np.nan, np.NaN, None, '']:
                numEmpty += 1
                continue
            tmpSents = body.split("。")
            tmpSents += title.split("。")#use both title and body
            
            tmpSents = [preprocess(item.strip()) for item in tmpSents]
            for s in tmpSents:
#                 s = s.encode('utf8')
                if len(s) < 3:
                    continue
                wordList = jieba.cut(s, cut_all=False)
                wordList = [item.encode('utf8') for item in wordList]
#                 print ' '.join(wordList)
                sents.append(wordList) 
        handle.close()
        print '%d empty new body found' % numEmpty   
         
        
    print '%d news sentences loaded' % (len(sents))
    
    print 'Learning phrase model'
    bigramConvertor = Phrases(sents)
    sents_bigram = bigramConvertor[sents]
    trigramConvertor = Phrases(sents_bigram)
    sents_trigram = trigramConvertor[sents_bigram]
    
    num_features = 100    # Word vector dimensionality                      
    min_word_count = 8   # Minimum word count                        
    num_workers = 2       # Number of threads to run in parallel
    contextWindow = 5          # Context window size                                                                                    
#     downsampling = 1000   # Downsample setting for frequent words
    
    # Initialize and train the model (this will take some time)
    
    
    print "Training model..."
#     model = word2vec.Word2Vec(sents, workers=num_workers, \
#                 size=num_features, min_count = min_word_count, \
#                 window = contextWindow, sample = downsampling, sg=0)
    model_unigram = word2vec.Word2Vec(sents, workers=num_workers, \
                size=num_features, min_count = min_word_count, window = contextWindow
                )
    model_bigram = word2vec.Word2Vec(sents_bigram, workers=num_workers, \
                size=num_features, min_count = min_word_count, window = contextWindow
                )
    model_trigram = word2vec.Word2Vec(sents_trigram, workers=num_workers, \
                size=num_features, min_count = min_word_count, window = contextWindow
                )
    # If you don't plan to train the model any further, calling 
    # init_sims will make the model much more memory-efficient.
#     model_unigram.init_sims(replace=True)
    
    # It can be helpful to create a meaningful model name and 
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name_unigram = "/home/lingandcs/workspace/FinancialNLP/data/model/2007_2015_word2vec_unigram.model"
    model_unigram.save(model_name_unigram)
    
    model_name_bigram = "/home/lingandcs/workspace/FinancialNLP/data/model/2007_2015_word2vec_bigram.model"
    model_bigram.save(model_name_bigram)
    
    model_name_trigram = "/home/lingandcs/workspace/FinancialNLP/data/model/2007_2015_word2vec_trigram.model"    
    model_trigram.save(model_name_trigram)
    
    
    print len(model_unigram.vocab), ' words learnt in uni model'
    print len(model_bigram.vocab), ' words learnt in bi model'
    print len(model_trigram.vocab), ' words learnt in tri model'
    
    print 'Testing'
    testWords = [u'医药', u'消费', u'机械', u'金融', u'能源', u'交通', u'运输', u'房产', 
                 u'地产', u'传媒', u'旅游',  u'化工', u'农林', u'贸易', u'信息', u'生物',  
                 u'健康', u'保险', u'餐饮', u'服装', u'纺织',  u'教育', u'服务', u'计算机', u'大数据',]
    
    for w in testWords:
        if w not in model_unigram.vocab:
            print w, 'is not in vocabulary'
            continue
        simWords = model_unigram.most_similar(w)
        simWords = [item[0] for item in simWords]
        print '\n\n Most similar words to:\t', w
        print ' '.join(simWords[:15])
    
    sys.exit()
#     for w in testWords:
#         if w not in model_bigram.vocab:
#             print w, 'is not in vocabulary'
#             continue
#         simWords = model_bigram.most_similar(w)
#         simWords = [item[0] for item in simWords]
#         print '\n\n Most similar words to:\t', w
#         print ' '.join(simWords[:10])
#     
#     for w in testWords:
#         if w not in model_trigram.vocab:
#             print w, 'is not in vocabulary'
#             continue
#         simWords = model_trigram.most_similar(w)
#         simWords = [item[0] for item in simWords]
#         print '\n\n Most similar words to:\t', w
#         print ' '.join(simWords[:10])
    
    
    
    
    vectorizer = TfidfVectorizer(max_features=150)
    
    sents = [' '.join(item) for item in sents]
    sents_bigram = [' '.join(item) for item in sents_bigram]
    sents_trigram = [' '.join(item) for item in sents_trigram]
    
    X = vectorizer.fit_transform(sents)
    print 'feature matrix shape\t', X.shape
    fns = vectorizer.get_feature_names()
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
        
        
        
    
    X = vectorizer.fit_transform(sents_bigram)
    print 'feature matrix shape\t', X.shape
    fns = vectorizer.get_feature_names()
    print '%d features got!' % len(fns)
    print X[0].shape
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
        
        
        
    
    X = vectorizer.fit_transform(sents_trigram)
    print 'feature matrix shape\t', X.shape
    fns = vectorizer.get_feature_names()
    print '%d features got!' % len(fns)
    print X[0].shape
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