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
# from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
import csv
import sys
import jieba

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
            tmpSents = [item.strip() for item in tmpSents]
            for s in tmpSents:
                wordList = jieba.cut(s, cut_all=False)
#                 print ' '.join(list(wordList))
                sents.append(list(wordList))
    print "%d sentences loaded" % len(sents)
    
    return sents
    
if __name__ == '__main__':
    
    filePathes = ["D:\\projects\\StockSentiment\\data\\20130101.csv", 
                  "D:\\projects\\StockSentiment\\data\\20130716.csv", 
                  "D:\\projects\\StockSentiment\\data\\20140101.csv", 
                  "D:\\projects\\StockSentiment\\data\\20140704.csv", 
                  "D:\\projects\\StockSentiment\\data\\20150101.csv"]
    sentences = getNewsSentences(filePathes)
    
#     for sent in sentences:
#         print sent
        
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print "Training Word2Vec model..."
    model = Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling, seed=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "D:\\projects\\StockSentiment\\data\\XWLB_2013-20150715.csv"
    model.save(model_name)

    model.doesnt_match("胡锦涛 李克强 一带一路 三个代表".split())
#     model.doesnt_match("paris berlin london austria".split())
    model.most_similar("李克强")
    model.most_similar("股市")
    model.most_similar("中国梦")