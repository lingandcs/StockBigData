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
    
    model_name = "/home/lingandcs/workspace/FinancialNLP/data/model/200features_4minwords_14context.model"
    w2v_model = Word2Vec.load(model_name)
    testWords = [u"能源", u"材料", u"工业", u"消费者", u"医疗", u"金融", u"信息技术", u"电信", u"公用",u"军工", u"互联网", u"体育", u"房地产", u"交通", u"保险", u"文化", u"铁路", u"航空", u"数据", u"李克强"]
    for w in testWords:
        if w not in w2v_model.vocab:
            print w, 'is not in vocabulary'
            continue
        simWords = w2v_model.most_similar(w, topn=50)
#         print len(simWords)
        simWords = [item[0] for item in simWords]
        print 'Most relevant words to:\t', w
        print ' '.join(simWords[:50])
        print '\n'
    