'''
Created on Apr 23, 2015
download foursquare data
@author: lingandcs
'''

import mysql.connector
import datetime
import time
import json
from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors import LinkExtractor
from scrapy.crawler import Crawler
from scrapy.contrib.loader import ItemLoader
from scrapy.contrib.loader.processor import Join, MapCompose, TakeFirst
from scrapy import log, signals, Spider, Item, Field
from scrapy.settings import Settings
from twisted.internet import reactor
from lxml import etree
from datetime import date
from dateutil.rrule import rrule, DAILY
import dateutil.parser
# from lxml.html.soupparser import fromstring
# from BSXPath import BSXPathEvaluator,XPathResult
import hashlib
import sys
from bs4 import BeautifulSoup
import urllib2
from test.test_isinstance import Child
import operator
from pandas.io.data import DataReader
import DataProcessor
import math
import matplotlib.pyplot as plt


apiPrefixReview_FourSquare = "https://api.foursquare.com/v2/venues/"
apiSurfixReview_FourSquare = "/tips?sort=recent&limit=500&client_id=G3RGK41B5M5QZWCPY23IQKXZDYK3LHT4TA2HDTAGFL4NFZTD&client_secret=4BF4J3JGWWBKLMEEWVHVZ2P1VFQGJXTBBLRZU0FVSMH4TAHM"

#return a list of triple(ER date, before data, after date); 2nd and 3rd elements are used to compute delta
def getERDates(ticker, cursor):
    cursor.execute("SELECT * FROM FinanceNLP.earning_calendar WHERE bizName = " + "\"" + ticker + "\"")
    result = cursor.fetchall()
    print "%d ERs loaded!"% len(result)
    ERs = []
    for row in result:
#         print row
        ERDate = row[4]
        dt = dateutil.parser.parse(str(row[4])).date()        
        beforeAfter = row[6].strip().lower()
        if beforeAfter.startswith("4:00") or beforeAfter.startswith("10:00 pm"):
            beforeAfter = 'After Market Close'.lower()
#         print dt, before, After
        if beforeAfter.startswith("before"):
#             print dt - datetime.timedelta(days=1), dt
            ERs.append([dt, dt - datetime.timedelta(days=1), dt])
        elif beforeAfter.startswith("after"):
#             print dt, dt + datetime.timedelta(days=1)
            ERs.append([dt, dt, dt + datetime.timedelta(days=1)])
        else:
            print "Fucking!!!" + beforeAfter
    ERs.sort(key=operator.itemgetter(1), reverse=False)

    return ERs
        
#divide time period by ER datae, then return cumulated review count
def getQuarterCumulated(erDates, dailyReviewCounts):
    er2ReviewQuarterlyCum = {}#key: data; value: counts
    #align first ER and first daily review count
    
    #ensure first ER date is later than first review count date
    i = 0    
    while erDates[i][0] <= dailyReviewCounts[0][0] and i < len(erDates):
        i += 1
    firstERDate = erDates[i][0]
#     print firstERDate
    #ensure first review date is later then 30 days before first ER date
    j = 0
    while dailyReviewCounts[j][0] < firstERDate - datetime.timedelta(days=90):
        j += 1
    firstReviewDate = dailyReviewCounts[j][0]
    
#     print firstERDate, firstReviewDate    
    
    while j < len(dailyReviewCounts):            
        if dailyReviewCounts[j][0] >= erDates[i][0]:
            i += 1#when date is out of current ER time window
        #initiate value in hash
        if erDates[i][0] not in er2ReviewQuarterlyCum:
            er2ReviewQuarterlyCum[erDates[i][0]] = 0            
#         print dailyReviewCounts[j][0], dailyReviewCounts[j][1], erDates[i][0]
        er2ReviewQuarterlyCum[erDates[i][0]] += dailyReviewCounts[j][1]
        j += 1
            
    return er2ReviewQuarterlyCum
             
#return the stock price change while ER
def getStockPriceSurprise(ticker, ERs):
    timewindowStockPrice = {}#key: year; value: weekly price
    stockDF = DataReader(ticker,  "google", "2007-01-01", datetime.datetime.today().date())
#     print stockDF
    ER2Surprise = {}#key:er date; value:price surprise
#     print stockDF.index
#     print stockDF.index[9]
#     print ERs
#     print ERs[0]
#     print stockDF

    for er in ERs:
#         print type(er[0])
        if er[0] >= datetime.datetime.today().date():
#             print er[0]
#             print 'too late ER!!!break'
            break
        if er[1] not in stockDF.index:
#             print er[1], 'not in stockDF, continue'
            continue
        before = stockDF.ix[er[1]]['Close']
        after = stockDF.ix[er[2]]['Close']
        ER2Surprise[er[0]] = (after - before)/before
#         print er[0], str(er[0])
#         print ER2Surprise[er[0]]

    return ER2Surprise
   
def getLossValue(quarterDeltas, ER2Surprise):
    lv = 0
    count = 0
    for er in quarterDeltas:
        dt = er[0]
        priceDiff = round(er[1],2)
        if dt in ER2Surprise:
            count += 1
#             print math.pow(ER2Surprise[dt]-priceDiff, 2), ER2Surprise[dt], priceDiff
            if ER2Surprise[dt]*priceDiff > 0:
                lv += 1
    return lv*1.0/count
    
    #how to normalize?
def getNormalizedQuarterlyReviewCount(quarterReviewCountsList):
    daylyAverageReviewCount = {}
    for i in xrange(len(quarterReviewCountsList)):
        daylyAverageReviewCount[i] = quarterReviewCountsList[i]
        if i == 0:
            #first quarter, no previous quarter to compare
#             print type(quarterDeltasList[i][1])
            daylyAverageReviewCount[i][1] = round(quarterReviewCountsList[i][1]*1.0/90, 3)#daily average count
        else:
#             print quarterDeltasList[i][0], quarterDeltasList[i-1][0]
            daysBtwER = quarterReviewCountsList[i][0] - quarterReviewCountsList[i-1][0]
#             print daysTimeWindow.days
            daylyAverageReviewCount[i][1] = round(quarterReviewCountsList[i][1]*1.0/daysBtwER.days, 3)
    
    return daylyAverageReviewCount
    
def getQuarterlyReviewCountDeltas(normalizedQUarterlyReviewCount):
    # calculate delta
    quarterDeltas = []
    for i in xrange(len(normalizedQUarterlyReviewCount)):
        if i == 0:
#             print type(quarterDeltasList[i][1])
            normalizedQUarterlyReviewCount[i][1] = 0
        else:
#             print daysTimeWindow.days
            if normalizedQUarterlyReviewCount[i-1][1] == 0:
                quarterDeltas.append([normalizedQUarterlyReviewCount[i][0], 0])
            else:
                quarterDeltas.append([normalizedQUarterlyReviewCount[i][0], (normalizedQUarterlyReviewCount[i][1]-normalizedQUarterlyReviewCount[i-1][1])/normalizedQUarterlyReviewCount[i-1][1]])
    
    return quarterDeltas
    
if __name__ == '__main__':
    ticker = "SBUX";
    bizFullName = "Starbucks"
    conn = mysql.connector.connect(user='lingandcs', password='sduonline',
                              host='107.170.18.102',
                              database='FinanceNLP')
    cursor_select = conn.cursor()
    erDates = getERDates(ticker, cursor_select)#read ER dates from my DB    
    dailyReviewCounts = DataProcessor.getDailyReviewsCount(bizFullName)#list of tuple(date, review count)    
    quarterReviewCounts = getQuarterCumulated(erDates, dailyReviewCounts)#review count over quarters, tuple list (ER date, previous quarter review count)    
    quarterReviewCountsList = [[item[0], item[1]] for item in quarterReviewCounts.items()]
    quarterReviewCountsList.sort(key = operator.itemgetter(0), reverse=False)
    normalizedQuarterlyReviewCount = getNormalizedQuarterlyReviewCount(quarterReviewCountsList)#TODY: other normalized method? or no normalize
    quarterDeltas = getQuarterlyReviewCountDeltas(normalizedQuarterlyReviewCount)#difference between quarters
    # now quarterDeltasList becomes daily average review count in each quarter    
    ER2Surprise = getStockPriceSurprise(ticker, erDates)#hash (date, price change)
#     print ER2Surprise

    lineReviewCount = []
    linePrice = []
    lineER = []
    print "Date\t", "ReviewCount Change\t", "Price Change\n", 
    for er in quarterDeltas:
        dt = er[0]
        reviewCountDiff = round(er[1],2)
        if dt in ER2Surprise:
            print dt, "\t%",reviewCountDiff, "\t%",round(ER2Surprise[dt]*100, 2)
            lineReviewCount.append(reviewCountDiff)
            linePrice.append(round(ER2Surprise[dt]*100, 2))
            lineER.append(dt)
        else:
            print dt, reviewCountDiff, None
             
    lossValue = getLossValue(quarterDeltas, ER2Surprise)
    print 'loss value:\t', lossValue
    print lineReviewCount
    print linePrice
    plt.plot(lineER, lineReviewCount, label='ReviewChange')
    plt.plot(lineER, linePrice, marker='o', linestyle='--', color='r', label='PriceChange')
    plt.xlabel('ER dates')
    plt.ylabel('Review & Price lines')
    plt.title(bizFullName + '  ER day price & review count change')
    plt.legend()
    plt.show()