#!/usr/bin/python
# -*- coding: utf-8 -*-
from operator import itemgetter


'''
Created on Jul 27, 2015
use clusters to replace words, then the same process
@author: lingandcs
'''
import pickle
import pandas as pd
import csv
import jieba
import re
from datetime import datetime
import csv
from dateutil.parser import parse
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.cluster import KMeans, MiniBatchKMeans

import pandas as pd
import numpy as np
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2
import sys
from gensim.models import Word2Vec
import time
import pickle



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


def loadXWLB(pathes, word2centroid):
    sents = []
    i = 0
    dt2xwlb = {}
    for path in pathes:
        handle = open(path, 'rb')
        csvReader = csv.reader(handle)
        for row in csvReader:
            if i%1000 == 0:
                print 'processing %d th news' % i
            i += 1
            dt = parse(row[0])
            title = row[1]
            body = row[2]
            sents = []
            tmpSents = body.split("。")
            tmpSents = [preprocess(item.strip()) for item in tmpSents]
            for s in tmpSents:
                if len(s) < 4:
                    continue
                wordList = jieba.cut(s, cut_all=False)
                centroidList = []
                for w in wordList:
                    if w in word2centroid:
#                         print w, word2centroid[w]
                        centroidList.append(str(word2centroid[w]))
                newSent = ' '.join(centroidList)
#                 print ' '.join(list(wordList))
                sents.append(newSent)
            sents = ' '.join(sents)#now sents is a string
            if dt not in dt2xwlb:
                dt2xwlb[dt] = ""
            dt2xwlb[dt] += sents + " "
    print "%d news loaded" % len(dt2xwlb)
    
    return dt2xwlb

def preprocessingPrice(dfPrices):
#     print dfPrices
    #key [Date     Open     High      Low    Close     amount       volumn]
    dailyPriceChange = []#element:[date, up/down/even]
    delta = 4
    for i,row in sectorPrice.iterrows():
        if i+delta >= dfPrices.shape[0]:
            break
        
        dt = parse(row['Date'])
        priceChange = 'even'
        currentPrice = row['Close']
#         print sectorPrice.irow(i)[0]
        newPrice = sectorPrice.irow(i+delta)[4]
#         print i,row['Close'], currentPrice, newPrice
        if (newPrice-currentPrice)/currentPrice > 0.04:
            priceChange = 'up'
        elif (newPrice-currentPrice)/currentPrice < -0.04:
            priceChange = 'down'
        
        dailyPriceChange.append([dt,priceChange])
#         print dailyPriceChange[-1]
    return dailyPriceChange

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

def getClusteringResult(w2v_model, modelExportPath):
    start = time.time()
    word_vectors = w2v_model.syn0
    num_clusters = word_vectors.shape[0] / 5
    kmeans_clustering = KMeans( n_clusters = num_clusters, n_jobs=2, verbose=True)
    pickle.dump(kmeans_clustering, open(modelExportPath, 'wb'))
    idx = kmeans_clustering.fit_predict( word_vectors )
    
    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print "Time taken for K Means clustering: ", elapsed, "seconds."
    
#     print w2v_model.index2word
    word2centroid_map = dict(zip( w2v_model.index2word, idx ))
#     print w2v_model.index2word
#     print idx
    
#     #build map from centroid to word list
    centroid2word = {}#key: centroid index; value: word list
    for item in word2centroid_map.items():
        w = item[0]
        idx_centroid = item[1]
        if idx_centroid not in centroid2word:
            centroid2word[idx_centroid] = []
        centroid2word[idx_centroid].append(w)
        
    #output clusters top N
    for cluster in xrange(0,20):
        print "\nCluster %d" % cluster
        words = []
        for i in xrange(0,len(word2centroid_map.values())):
            if( word2centroid_map.values()[i] == cluster ):
                words.append(word2centroid_map.keys()[i])
        print ' '.join(words)
    
    return word2centroid_map, centroid2word
    
    
if __name__ == '__main__':
    
    model_name = "/home/lingandcs/workspace/FinancialNLP/data/model/200features_4minwords_14context.model"
    clusteringModelExportPath = "/home/lingandcs/workspace/FinancialNLP/data/model/clustering_kmeans.model"
    w2v_model = Word2Vec.load(model_name)
    print 'word 2 vector model loaded'
    
    word2centroid_map, centroid2word = getClusteringResult(w2v_model, clusteringModelExportPath)
    
    sectorPrice = pd.read_pickle('/home/lingandcs/workspace/FinancialNLP/data/Equity price data/HS300FN.pkl')
#     print df['Close']
    
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
                "/home/lingandcs/workspace/FinancialNLP/data/20100101_20100505.csv",
                "/home/lingandcs/workspace/FinancialNLP/data/20100506_20101231.csv",
                "/home/lingandcs/workspace/FinancialNLP/data/20090101_20090625.csv",
                "/home/lingandcs/workspace/FinancialNLP/data/20090626_20091231.csv",
                "/home/lingandcs/workspace/FinancialNLP/data/20080101_20080630.csv",
                "/home/lingandcs/workspace/FinancialNLP/data/20080701_20081231.csv",
                "/home/lingandcs/workspace/FinancialNLP/data/20070101_20070630.csv",
                "/home/lingandcs/workspace/FinancialNLP/data/20070701_20071231.csv",
                  ]
    dt2xwlb = loadXWLB(filePathes, word2centroid_map)
#     for key in dt2xwlb.keys():
#         print key, dt2xwlb[key]
    dailySectorPrice = preprocessingPrice(sectorPrice)
    
    startDT = parse('20070101')
    endDT = parse('20150713')
    
    ALL = []
    for pair in dailySectorPrice:
        dt = pair[0]
        if dt < startDT or dt > endDT:
            continue
        if dt not in dt2xwlb:
            print 'fuck ', dt
            continue
        
        priceChange = pair[1]        
        
        ALL.append([dt2xwlb[dt], priceChange])
        
#     random.seed(500)
#     random.shuffle(ALL)
    
    X = [item[0] for item in ALL]
    Y = [item[1] for item in ALL]
    
    
    
#     print X
#     print Y
    lengthTrain = int(len(ALL)*0.6)
    ALL_train = ALL[:lengthTrain]
    ALL_test = ALL[lengthTrain:]
    X_train = X[:lengthTrain]
    X_test = X[lengthTrain:]
    Y_train = Y[:lengthTrain]
    Y_test = Y[lengthTrain:]

    
    
    
    vectorizer = CountVectorizer(tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000, ngram_range =(1,2), binary=True)
#     vectorizer = TfidfVectorizer(max_features = 15000, ngram_range =(1,3))    
    
    train_data_features = vectorizer.fit_transform(X_train)

    feature_names_original = vectorizer.get_feature_names()#from original feature set
    
    
    ch2 = SelectKBest(chi2, k=3500)
    train_data_features = ch2.fit_transform(train_data_features, Y_train)
    feature_names = [feature_names_original[idx] for idx in ch2.get_support(indices=True)]

    print len(feature_names), 'new features loaded after feature selection'
#     print feature_names

        
#     sys.exit()  
    
    
    
    train_data_features = train_data_features.toarray()
    
    # Initialize a Random Forest classifier with 100 trees
#     clf = RandomForestClassifier(n_estimators = 50, n_jobs = 2)
#     clf = BernoulliNB()
#     clf = GradientBoostingClassifier(verbose=True, learning_rate=0.001, n_estimators=1000, max_depth=4, max_features='auto')
    clf = MultinomialNB()
#     clf = SGDClassifier()
#     clf = linear_model.logistic()
#     clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
#     clf = SVC(kernel='linear')    
    
    clf = clf.fit( train_data_features, Y_train)

    test_data_features = vectorizer.transform(X_test)    
    test_data_features = ch2.transform(test_data_features)
    test_data_features = test_data_features.toarray()
    
    print "Predicting test labels...\n"
    y_pred = clf.predict(test_data_features)
    print metrics.classification_report(Y_test, y_pred)
    
    print 'output feature importance'    
    classLabels = clf.classes_
    
    for i, classLabel in enumerate(classLabels):
        print '\n\nFeature names for class \t', classLabel
        coefs_with_fns = sorted(zip(clf.coef_[i], feature_names), reverse=True)
        for coef, fn in coefs_with_fns[:20]:
            print coef, fn
            fs = fn.split()
            for f in fs:
                print ' '.join(centroid2word[int(f)])
    