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

    
    
if __name__ == '__main__':
    
    model_name = "/home/lingandcs/workspace/FinancialNLP/data/model/2007_2015_word2vec_bigram.model"
    w2v_model = Word2Vec.load(model_name)
    testWords = ['医疗', '消费', '机械', '金融', '能源', '交通', '运输', '房产', 
                 '地产', '传媒', '旅游',  '化工', '农林', '贸易', '信息', '生物',  
                 '健康', '保险', '餐饮', '服装', '纺织',  '教育', '服务', '计算机', '大数据', "汽车", "军工"]
    for w in testWords:
        w = w.decode('utf-8')
        if w not in w2v_model.vocab:
            print w, 'is not in vocabulary'
            continue
        simWords = w2v_model.most_similar(w, topn=50)
#         print len(simWords)
        simWords = [item[0] for item in simWords]
        print 'Most relevant words to:\t', w
        print ' '.join(simWords[:20])
        print '\r'
    