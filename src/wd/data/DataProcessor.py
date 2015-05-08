'''
Created on Apr 23, 2015
download foursquare data
@author: lingandcs
'''

import mysql.connector
from datetime import datetime, timedelta, date
import time
import json
import urllib2
import sys
import json
import operator
import matplotlib.pyplot as plt
import random
from pandas.io.data import DataReader
import dateutil.parser
import math
import scipy
import numpy

def getHistogram(hist):
    sorted_hist = sorted(hist.items(), key=operator.itemgetter(0), reverse=True)
    numReviews = [pair[0] for pair in sorted_hist]# x-axis
    numbizs = [pair[1] for pair in sorted_hist]# y-axis
    plt.bar(numReviews, numbizs)
    plt.show()
    
def getTimewindowReviewsCount(bizName):
    conn = mysql.connector.connect(user='lingandcs', password='sduonline',
                              host='107.170.18.102',
                              database='goodfoodDB')
    query_select = "SELECT * FROM goodfoodDB.FinanceNLP_review_FS WHERE bizName = '" + bizName + "'"
    reviews = None
    
    try:
        cursor = conn.cursor()        
        #Hard code biz name, to be changed
        cursor.execute(query_select)
        
        reviews = cursor.fetchall()
        print "%d reviews loaded!"% len(reviews)
        
        hist = {}#key: number of review in certain biz; value: number of biz
        for row in reviews:
            count = row[0]
            if count not in hist:
                hist[count] = 0
            hist[count] += 1        
#         getHistogram(hist)
    finally:
        conn.close()
        
    timewindowReviewCount = {}#key:year; value: weekly number of reviews in each year
        
    for review in reviews:
#         print review
        dt = review[2]
#         print dt
#         print type(dt)
        year = dt.isocalendar()[0]
        week = dt.isocalendar()[1]
#         print type(year)
#         print week
        year_week = str(year) + str(week)
        if year not in timewindowReviewCount:
            timewindowReviewCount[year] = {}
        if week not in timewindowReviewCount[year]:
            timewindowReviewCount[year][week] = 0
        timewindowReviewCount[year][week] += 1
        
    return timewindowReviewCount
        
def getDailyReviewsCount(bizName):
    conn = mysql.connector.connect(user='lingandcs', password='sduonline',
                              host='107.170.18.102',
                              database='goodfoodDB')
    query_select = "SELECT * FROM goodfoodDB.FinanceNLP_review_FS WHERE bizName = '" + bizName + "'"
    reviews = None
    
    try:
        cursor = conn.cursor()        
        #Hard code biz name, to be changed
        cursor.execute(query_select)
        
        reviews = cursor.fetchall()
        print "%d reviews loaded!"% len(reviews)
        
        hist = {}#key: number of review in certain biz; value: number of biz
        for row in reviews:
            count = row[0]
            if count not in hist:
                hist[count] = 0
            hist[count] += 1        
#         getHistogram(hist)
    finally:
        conn.close()
        
    timewindowReviewCount = {}#key:year; value: weekly number of reviews in each year
        
    for review in reviews:
#         print review
        dt = review[2]
#         print type(year)
#         print week
        if dt not in timewindowReviewCount:
            timewindowReviewCount[dt] = 0
        timewindowReviewCount[dt] += 1
        
    timewindowReviewCountList = timewindowReviewCount.items()
    
    timewindowReviewCountList.sort(key = operator.itemgetter(0), reverse = False)
#     for item in timewindowReviewCountList:
#         print item
    return timewindowReviewCountList
        
def getTimewindowStockPrice(ticker):
    timewindowStockPrice = {}#key: year; value: weekly price
    stockDF = DataReader(ticker,  "google", "2009-01-01", datetime.today().date())
#     print stockDF
    for idx, row in stockDF.iterrows():
#         print row[0], row['Close']
#         print datetime.fromtimestamp(idx)
#         print str(idx)
        dt = dateutil.parser.parse(str(idx)).date()
        year = dt.isocalendar()[0]
        week = dt.isocalendar()[1]
        price = row['Close']
        print year, week, price
        if year not in timewindowStockPrice:
            timewindowStockPrice[year] = {}
        if week not in timewindowStockPrice[year]:
            timewindowStockPrice[year][week] = []
#         print row['Close']
        timewindowStockPrice[year][week].append(price)
        
    #normalized weekly price
    for year in timewindowStockPrice.keys():
        for week in timewindowStockPrice[year].keys():
            timewindowStockPrice[year][week] = scipy.mean(timewindowStockPrice[year][week])
    
#     for year in timewindowStockPrice.keys():
#         print timewindowStockPrice[year]
    return timewindowStockPrice
    
def normalize(series):
    m = scipy.median(series)
    newSeries = [item/m for item in series]
    print newSeries
    return newSeries
    
if __name__ == '__main__':
    
    result = DataReader("TWTR",  "google", "2015-04-28", "2015-04-28")
#     print result
    
    
    weeklyReviewCount = getTimewindowReviewsCount("Shake Shack")
    sys.exit()
    
    
    weeklyStockPrice = getTimewindowStockPrice("SHAK")
    
    yearWeeklyReviewCount = weeklyReviewCount[2015]
    yearWeeklyStockPrice = weeklyStockPrice[2015]
#     print yearWeeklyReviewCount
#     print len(yearWeeklyReviewCount)
    
    sorted_review = sorted(yearWeeklyReviewCount.items(), key=operator.itemgetter(0), reverse=False)
    sorted_price = sorted(yearWeeklyStockPrice.items(), key=operator.itemgetter(0), reverse=False)
    sorted_review = [pair[1] for pair in sorted_review]
    sorted_price = [pair[1] for pair in sorted_price]
    sorted_review = normalize(sorted_review)
    sorted_price = normalize(sorted_price)
    
#     print sorted_review
#     print sorted_price
    
    plt.plot(range(len(sorted_review)), sorted_review, 'k', label="review counts")
    plt.plot(range(len(sorted_price)), sorted_price, '-', label="stock price")
    plt.legend()
    plt.gca().xaxis.grid(True)
    plt.gca().yaxis.grid(True)
    plt.xlabel("nth week")
    plt.ylabel("Review count and Price value")
    plt.title("Trend of review counts and stock price of Shake Shack 2015")
    
    plt.plot(range(len(yearWeeklyReviewCount)), yearWeeklyReviewCount, yearWeeklyStockPrice)
    plt.axis()
    plt.show()
    
    
#     for i, row in enumerate(yearWeeklyReviewCount):
#         print i+1, yearWeeklyReviewCount[i+1], yearWeeklyStockPrice[i+1]
#     for year in weeklyReviewCount:
#         print 'year\t:', year
#         print weeklyReviewCount[year]
    
#     print datetime.today().date()
    
    