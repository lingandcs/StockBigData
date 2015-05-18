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

import DataProcessor

apiPrefixReview_FourSquare = "https://api.foursquare.com/v2/venues/"
apiSurfixReview_FourSquare = "/tips?sort=recent&limit=500&client_id=G3RGK41B5M5QZWCPY23IQKXZDYK3LHT4TA2HDTAGFL4NFZTD&client_secret=4BF4J3JGWWBKLMEEWVHVZ2P1VFQGJXTBBLRZU0FVSMH4TAHM"

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
#         print dt, beforeAfter
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
        
#return quarterly cumulation review count
def getQuarterDeltas(erDates, dailyReviewCounts):
    er2ReviewQuarterlyCum = {}
    
    #align first ER and first daily review count
    #ensure first ER date is after 30 days after first review date
    i = 0    
    while erDates[i][0] <= dailyReviewCounts[0][0] and i < len(erDates):
        i += 1
    firstERDate = erDates[i][0]
    
    #ensure first review date is later then 30 days before first ER date
    j = 0
    while dailyReviewCounts[j][0] < firstERDate - datetime.timedelta(days=90):
        j += 1
    firstReviewDate = dailyReviewCounts[j][0]
    
    print firstERDate, firstReviewDate
    
    
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
                
if __name__ == '__main__':
    conn = mysql.connector.connect(user='lingandcs', password='sduonline',
                              host='107.170.18.102',
                              database='FinanceNLP')
    cursor_select = conn.cursor()
    erDates = getERDates('CMG', cursor_select)
    dailyReviewCounts = DataProcessor.getDailyReviewsCount("Chipotle")    
    quarterDeltas = getQuarterDeltas(erDates, dailyReviewCounts)
    quarterDeltasList = [[item[0], item[1]] for item in quarterDeltas.items()]
    quarterDeltasList.sort(key = operator.itemgetter(0), reverse=False)
    
    #normalize?
    for i in xrange(len(quarterDeltasList)):
        if i == 0:
#             print type(quarterDeltasList[i][1])
            quarterDeltasList[i][1] = round(quarterDeltasList[i][1]*1.0/90, 3)
        else:
#             print quarterDeltasList[i][0], quarterDeltasList[i-1][0]
            daysTimeWindow = quarterDeltasList[i][0] - quarterDeltasList[i-1][0]
#             print daysTimeWindow.days
            quarterDeltasList[i][1] = round(quarterDeltasList[i][1]*1.0/daysTimeWindow.days, 3)
        
    #calculate delta
    for i in xrange(len(quarterDeltasList)):
        if i == 0:
#             print type(quarterDeltasList[i][1])
            quarterDeltasList[i][1] = 0
        else:
#             print daysTimeWindow.days
            quarterDeltasList[i][1] = quarterDeltasList[i][1] - quarterDeltasList[i-1][1]
            
    print quarterDeltasList
#     for er in erDates:
#         print er
#     for item in dailyReviewCounts:
#         print item