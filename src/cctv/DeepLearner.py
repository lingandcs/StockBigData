#!/usr/bin/python
# -*- coding: utf-8 -*-


'''
Created on Jul 19, 2015
train neural network language model
@author: lingandcs
'''

import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
import csv
import sys
import jieba
import glob
from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time


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
    doc = doc.replace('[', '')
    doc = doc.replace(']', '')
    return doc


def getNewsSentences(pathes):
    sents = []
    i = 0
    for path in pathes:
        handle = open(path, 'rb')
        csvReader = csv.reader(handle)
        for row in csvReader:
            if i%1000 == 0:
                print 'processing %d th news' % i
            i += 1
            tmpSents = row[2].split("。")
#             tmpSents = row[1].split("。")
            tmpSents = [preprocess(item.strip()) for item in tmpSents]
            for s in tmpSents:
                if len(s) < 4:
                    continue
                wordList = jieba.cut(s, cut_all=False)
#                 print ' '.join(list(wordList))
                sents.append(list(wordList))
    print "%d sentences loaded" % len(sents)
    
    return sents
    
if __name__ == '__main__':
    
    filePathes = ["/home/lingandcs/workspace/FinancialNLP/data/20150101_20150713.csv", 
                  "/home/lingandcs/workspace/FinancialNLP/data/20130101_20130715.csv", 
                  "/home/lingandcs/workspace/FinancialNLP/data/20140101_20140703.csv", 
                  "/home/lingandcs/workspace/FinancialNLP/data/20130716_20131231.csv", 
                  "/home/lingandcs/workspace/FinancialNLP/data/20110101_20110405.csv",
                  "/home/lingandcs/workspace/FinancialNLP/data/20140704_20141231.csv",
                  "/home/lingandcs/workspace/FinancialNLP/data/20150101_20150713.csv",
                  "/home/lingandcs/workspace/FinancialNLP/data/20120701_20121231.csv",
                  "/home/lingandcs/workspace/FinancialNLP/data/20110406_20110630.csv",
                  "/home/lingandcs/workspace/FinancialNLP/data/20120101_20120630.csv",
                  "/home/lingandcs/workspace/FinancialNLP/data/20110701_20111231.csv",
                  ]
    sentences = getNewsSentences(filePathes)    
    
    
#     for sent in sentences:
#         print sent
        
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 100    # Word vector dimensionality
    min_word_count = 10   # Minimum word count
    num_workers = 2       # Number of threads to run in parallel
    context = 5         # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print "Training Word2Vec model..."
#     model = Word2Vec(sentences, workers=num_workers, \
#                 size=num_features, min_count = min_word_count, \
#                 window = context, sample = downsampling, seed=1)
    model = Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                )
    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "/home/lingandcs/workspace/FinancialNLP/data/XWLB_2013-20150715.csv"
    model.save(model_name)

    testWords = [u'习近平', u'李克强', u'钓鱼岛', u'日本', u'朝鲜', u'台湾', u'奥巴马', u'科学', u'发展观', u'科学发展观', u"新疆", u"经济", u"互联网", u"创业"]
    for w in testWords:
        if w not in model.vocab:
            print w, 'is not in vocabulary'
            continue
        simWords = model.most_similar(w)
        simWords = [item[0] for item in simWords]
        print '\n\n Most similar words to:\t', w
        print ' '.join(simWords[:10])    
    
    wordsInVocab = model.index2word
    
    print len(wordsInVocab), ' words in vocab'
    
    featureVecs = []
    
    for w in wordsInVocab:
#         featureVecs = np.add(featureVecs,model[w].transpose())
        featureVecs.append(list(model[w]))
    
    featureVecs = np.array(featureVecs)
    print 'feature matrix shape:\t', featureVecs.shape
    
    t0 = time()  
     
    k = 200
    km = KMeans(n_clusters=k, init='k-means++', max_iter=10000, n_init=1,
                verbose=True)
    t0 = time()
    
    km.fit(featureVecs)
    print("done in %0.3fs" % (time() - t0))
#     print featureVecs

    
    wordClusters = {}#key:cluster label; value:list of words
    #index i is used in both vocabulary and labels
    for i, label in enumerate(km.labels_):        
        if label not in wordClusters:
            wordClusters[label] = []
        wordClusters[label].append(wordsInVocab[i])
    pairs = wordClusters.items()
    pairs.sort(key = lambda x:len(x[1]))
    
    print 'output word clusters:\t'
    for item in pairs[::-1]:
        print item[0], ' '.join(item[1][:])
    
#     for item in wordClusters.items():
#         print item[0], ' '.join(item[1])