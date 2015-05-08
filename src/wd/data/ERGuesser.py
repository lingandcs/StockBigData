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
        print row
        ERDate = row[4]
        dt = dateutil.parser.parse(str(row[4])).date()        
        beforeAfter = row[6].strip().lower()
        if beforeAfter.startswith("4:00") or beforeAfter.startswith("10:00 pm"):
            beforeAfter = 'After Market Close'.lower()
        print dt, beforeAfter
        if beforeAfter.startswith("before"):
            print dt - datetime.timedelta(days=1), dt
            ERs.append([dt, dt - datetime.timedelta(days=1), dt])
        elif beforeAfter.startswith("after"):
            print dt, dt + datetime.timedelta(days=1)
            ERs.append([dt, dt, dt + datetime.timedelta(days=1)])
        else:
            print "Fucking!!!" + beforeAfter
    ERs.sort(key=operator.itemgetter(1), reverse=False)
    
    return ERs
        
def getQuarterDeltas(erDates, dailyReviewCounts):
    er2ReviewCum = {}
    i = 0
    erDate = erDates[i][0]
#     while(i < len(erDates)):
#     for i,erDateInfo in enumerate(erDates):
    
    for j,dReviewInfo in enumerate(dailyReviewCounts):
        erDate = erDates[i][0]
        if erDate not in er2ReviewCum:
            er2ReviewCum[erDate] = 0
        d = dReviewInfo[0]
        rCount = dReviewInfo[1]
        if d < erDate:
            er2ReviewCum[erDate] += rCount
        else:
            i += 1
            if i >= len(erDates):
                break
            
    return er2ReviewCum
                
if __name__ == '__main__':
    conn = mysql.connector.connect(user='lingandcs', password='sduonline',
                              host='107.170.18.102',
                              database='FinanceNLP')
    cursor_select = conn.cursor()
    erDates = getERDates('CMG', cursor_select)
    dailyReviewCounts = DataProcessor.getDailyReviewsCount("Chipotle")    
    quarterDeltas = getQuarterDeltas(erDates, dailyReviewCounts)
    quarterDeltasList = quarterDeltas.items()
    quarterDeltasList.sort(key = operator.itemgetter(0), reverse=False)
    print quarterDeltasList
#     for er in erDates:
#         print er
#     for item in dailyReviewCounts:
#         print item