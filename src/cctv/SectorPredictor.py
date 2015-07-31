#!/usr/bin/python
# -*- coding: utf-8 -*-


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

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics


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


def loadXWLB(pathes):
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
                newSent = ' '.join(wordList)
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
    delta = 2
    for i,row in sectorPrice.iterrows():
        if i+delta >= dfPrices.shape[0]:
            break
        
        dt = parse(row['Date'])
        priceChange = 'even'
        currentPrice = row['Close']
#         print sectorPrice.irow(i)[0]
        newPrice = sectorPrice.irow(i+delta)[4]
#         print i,row['Close'], currentPrice, newPrice
        if (newPrice-currentPrice)/currentPrice > 0.02:
            priceChange = 'up'
        elif (newPrice-currentPrice)/currentPrice < -0.02:
            priceChange = 'down'
        
        dailyPriceChange.append([dt,priceChange])
#         print dailyPriceChange[-1]
    return dailyPriceChange

if __name__ == '__main__':
    
    sectorPrice = pd.read_pickle('/home/lingandcs/workspace/FinancialNLP/data/Equity price data/HS300FN.pkl')
#     print df['Close']
    print type(sectorPrice['Close'][2554])
    
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
    dt2xwlb = loadXWLB(filePathes)
#     for key in dt2xwlb.keys():
#         print key, dt2xwlb[key]
    dailySectorPrice = preprocessingPrice(sectorPrice)
    
    startDT = parse('20110101')
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
                             max_features = 5000, ngram_range =(1,3), binary=True)
#     vectorizer = TfidfVectorizer(max_features = 5000, ngram_range =(1,3))

    train_data_features = vectorizer.fit_transform(X_train)
    train_data_features = vectorizer.fit_transform(X_train)

    train_data_features = train_data_features.toarray()

    # Initialize a Random Forest classifier with 100 trees
#     clf = RandomForestClassifier(n_estimators = 200, n_jobs = 2)
    clf = BernoulliNB()
    clf = GradientBoostingClassifier(verbose=True, learning_rate=0.001, n_estimators=10000, max_depth=4, max_features='auto')
    clf = MultinomialNB()
    clf = SGDClassifier()
    
    
    clf = clf.fit( train_data_features, Y_train)

    test_data_features = vectorizer.transform(X_test)
    test_data_features = vectorizer.fit_transform(X_test)
    test_data_features = test_data_features.toarray()

    print "Predicting test labels...\n"
    y_pred = clf.predict(test_data_features)
    print metrics.classification_report(Y_test, y_pred)
    
    print 'output feature importance'
    vocab = vectorizer.get_feature_names()
#     for w in vocab:
#         print w
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[2], feature_names), reverse=True)

    
#     for coef_1, fn_1 in coefs_with_fns[:1000]:
#         print coef_1, fn_1
        
    classLabels = clf.classes_
    for i, classLabel in enumerate(classLabels):
        print 'top features for \t', classLabel
        topN = np.argsort(clf.coef_[i])[:30]
        print ("%s: %s" % (classLabel, " ".join(feature_names[j] for j in topN)))