#!/usr/bin/python
# -*- coding: utf-8 -*-
# from operator import itemgetter


'''
Created on Jul 27, 2015

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

import pandas as pd
import numpy as np
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2
import sys

def preprocess(doc):
#     print doc
    doc = doc.replace('视频', '')
    doc = doc.replace('新闻联播', '')
    doc = doc.replace('本台消息', '')
    doc = doc.replace('联播', '')
    doc = doc.replace('快讯', '')
    doc = doc.replace('联播', '')
    doc = doc.replace('联播', '')
#     doc = doc.replace('新闻联播', '')
#     doc = doc.replace('新闻联播', '')
#     doc = doc.replace('【', '')
#     doc = doc.replace('】', '')
    doc = re.sub(r'\[\]','',doc)
    doc = re.sub(r'【|】','',doc)
    doc = re.sub(r'\d+','DIGIT',doc)
    return doc


def loadXWLB(pathes):
    sents = []
    i = 0
    dt2xwlb = {}
    for path in pathes:
        handle = open(path, 'rb')
        csvReader = csv.reader(handle)
        for row in csvReader:
            if i%10000 == 0:
                print 'processing %d th news' % i
            i += 1
            dt = parse(row[0])
            title = row[1]
            body = row[2]
            sents = []
            tmpSents = title.split("。")
            tmpSents = [preprocess(item.strip()) for item in tmpSents]
            for s in tmpSents:
                if len(s) < 4:
                    continue
                wordList = jieba.cut(s, cut_all=False)
                newSent = ' '.join(wordList)
#                 print ' '.join(list(wordList))
                sents.append(newSent)
            sents = ' '.join(sents)#now sents is a string
            if dt not in dt2xwlb:
                dt2xwlb[dt] = ""
            dt2xwlb[dt] += sents + " "
    print "%d daily news loaded" % len(dt2xwlb)
    
    return dt2xwlb

def preprocessingPrice(dfPrices):
#     print dfPrices
    #key [Date     Open     High      Low    Close     amount       volumn]
    dailyPriceChange = []#element:[date, up/down/even]
    delta = 4
    numPrice = 0
    for i,row in sectorPrice.iterrows():
        numPrice += 1
        if i+delta >= dfPrices.shape[0]:
            break
        
        dt = parse(row['Date'])
#         print dt
        priceChange = 'even'
        currentPrice = row['Close']
#         print sectorPrice.irow(i)[0]
        newPrice = sectorPrice.irow(i+delta)[4]
#         print i,row['Close'], currentPrice, newPrice
        if (newPrice-currentPrice)/currentPrice > 0.03:
            priceChange = 'up'
        elif (newPrice-currentPrice)/currentPrice < -0.03:
            priceChange = 'down'
        
        dailyPriceChange.append([dt,priceChange])
#         print dailyPriceChange[-1]
    print "%d price loaded, no price in weekend!!!" % numPrice
    return dailyPriceChange

if __name__ == '__main__':
    
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
    dt2xwlb = loadXWLB(filePathes)
#     for key in dt2xwlb.keys():
#         print key, dt2xwlb[key]
    dailySectorPrice = preprocessingPrice(sectorPrice)
    
    startDT = parse('20070101')
    endDT = parse('20150713')
    
    ALL = []
    numMissedPrice = 0
    for pair in dailySectorPrice:
        dt = pair[0]
        if dt < startDT or dt > endDT:
            continue
        if dt not in dt2xwlb:
#             print 'fuck ', dt
            numMissedPrice += 1
            continue
        
        priceChange = pair[1]        
        
        ALL.append([dt2xwlb[dt], priceChange])
    
    print '%d price are not found in XWLB' % numMissedPrice
    print len(ALL), ' data points loaded in total'
#     random.seed(100)
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
                             max_features = 50000, ngram_range =(1,2), binary=True)
#     vectorizer = TfidfVectorizer(max_features = 50000, ngram_range =(1,2))    
    
    train_data_features = vectorizer.fit_transform(X_train)
    train_data_features = vectorizer.fit_transform(X_train)

    feature_names_original = vectorizer.get_feature_names()#from original feature set
    
    
    ch2 = SelectKBest(chi2, k=1000)
    train_data_features = ch2.fit_transform(train_data_features, Y_train)
    feature_names = [feature_names_original[idx] for idx in ch2.get_support(indices=True)]

    print len(feature_names), 'new features loaded after feature selection'
#     print feature_names

        
#     sys.exit()
    
    
    
    
    
    
    train_data_features = train_data_features.toarray()
    
    # Initialize a Random Forest classifier with 100 trees
    clf = RandomForestClassifier(n_estimators = 50, n_jobs = 2)
#     clf = BernoulliNB()
    clf = GradientBoostingClassifier(verbose=True, learning_rate=0.001, n_estimators=1000, max_depth=4, max_features='auto')
    clf = MultinomialNB()
#     clf = SGDClassifier()
#     clf = linear_model.logistic()
#     clf = SVC(kernel='linear')
    
    
    clf = clf.fit( train_data_features, Y_train)

    test_data_features = vectorizer.transform(X_test)
    test_data_features = vectorizer.fit_transform(X_test)
    
    test_data_features = ch2.transform(test_data_features)
    test_data_features = test_data_features.toarray()
    
    print "Predicting test labels...\n"
    y_pred = clf.predict(test_data_features)
    print metrics.classification_report(Y_test, y_pred)
    
    print 'output feature importance'
    
    classLabels = clf.classes_
    
    for i, classLabel in enumerate(classLabels):
        print 'Feature names for class \t', classLabel
        coefs_with_fns = sorted(zip(clf.coef_[i], feature_names), reverse=True)
        for fn, coef in coefs_with_fns[:10]:
            print fn, coef
    