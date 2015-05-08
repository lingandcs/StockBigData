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
# from lxml.html.soupparser import fromstring
# from BSXPath import BSXPathEvaluator,XPathResult
import hashlib

from bs4 import BeautifulSoup
import urllib2
from test.test_isinstance import Child

apiPrefixReview_FourSquare = "https://api.foursquare.com/v2/venues/"
apiSurfixReview_FourSquare = "/tips?sort=recent&limit=500&client_id=G3RGK41B5M5QZWCPY23IQKXZDYK3LHT4TA2HDTAGFL4NFZTD&client_secret=4BF4J3JGWWBKLMEEWVHVZ2P1VFQGJXTBBLRZU0FVSMH4TAHM"

#adapt to different TR format and get column index for company, symbol, and time
def getInfoIndex(tds):
    infoDict = {}
    for i, td in enumerate(tds):
        if td.string == None:
            continue
        if td.string.lower() in ['company']:
            infoDict["company"] = i
        if td.string.lower() in ['symbol']:
            infoDict["symbol"] = i
        if td.string.lower() in ['time']:
            infoDict["time"] = i
#         print td.string
        
    return infoDict

def getSingleDayEarnings(u):
    earnings = []
    
    content = None
    try:
        content = urllib2.urlopen(u).read()
#         print content
    except urllib2.HTTPError:
        print 'fucking valid page!!!'
        return None
#     print content
    soup = BeautifulSoup(content)
    trs = soup.find_all('tr')
#     print len(list(trs))
    infoIndex = None
    for tr in trs:
        tdList = list(tr.children)        
        if len(tdList) < 3 or tdList[0].string == None:
            continue
#         print len(tdList)
        elif tdList[0].string.lower() in ["company"]:
            #process title rows
            infoIndex = getInfoIndex(tdList)
            continue
        if infoIndex == None or infoIndex == {}:
            continue
#         print tdList
        tdTexts = [unicode(td.string) for td in tdList]
        companyName = tdTexts[infoIndex['company']]
        symbol = tdTexts[infoIndex['symbol']]
        beforeAfter = tdTexts[infoIndex['time']]
        
        earnings.append([symbol, companyName, symbol, beforeAfter])
#         print "%s\t%s\t%s" % (companyName, symbol, beforeAfter)
        
    return earnings
    
        
def getAllDaysEarnings(startDay, endDay):
    conn = mysql.connector.connect(user='lingandcs', password='sduonline',
                              host='107.170.18.102',
                              database='FinanceNLP')
    query_insert = "INSERT INTO FinanceNLP.earning_calendar (bizName, fullName, ticker, earningDate, beforeAfter, checksum) "  \
                    "VALUES (%s,%s,%s,%s,%s,%s)"
    cursor_insert = conn.cursor()
    
    countEarning = 0
    countException = 0
    
    for dt in rrule(DAILY, dtstart=startDay, until=endDay):
        dtStr = dt.strftime("%Y%m%d")
        u = "http://biz.yahoo.com/research/earncal/" + dtStr + ".html"
#         print '--------------------------------------------'
        print u
        individualDayEarnings = getSingleDayEarnings(u)
        if individualDayEarnings == None:
            continue
#         print dt.strftime("%Y-%m-%d")
#         print individualDayEarnings
        
        for earning in individualDayEarnings:
            try:
#                 print len(earning)
#                 print earning
                try:
                    checksum = hashlib.md5(earning[1]+dt.strftime("%Y-%m-%d")+earning[3]).hexdigest()
                except UnicodeEncodeError:
                    continue
#                 print type(earning[0]), type(earning[1]), type(earning[2]), type(dt.strftime("%Y-%m-%d")), type(earning[3]), type(checksum)
                cursor_insert.execute(query_insert, (earning[0], earning[1], earning[2], dt.strftime("%Y-%m-%d"), earning[3], checksum))
                conn.commit()
                countEarning += 1
                if countEarning%1000 == 0:
                    print '%d reviews inserted'% countEarning
            except mysql.connector.errors.IntegrityError:
#                 print 'fuck, INSERT failure!!!'
                countException += 1
                pass
    
    print "%d earnings extracted" % (countEarning)
    print "%d exceptions raised" % (countException)
    conn.close()
        
        
if __name__ == '__main__':

    url = "http://biz.yahoo.com/research/earncal/20150508.html"
#     getContent(url)
    start = datetime.date(2010,01,01)
    end = datetime.date(2015,12,31)
    getAllDaysEarnings(start, end)