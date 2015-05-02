'''
Created on Apr 23, 2015
download foursquare data
@author: lingandcs
'''

import mysql.connector
from datetime import datetime
import time
import json
import urllib2
import sys
import json
import scrapy
from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors import LinkExtractor

apiPrefixReview_FourSquare = "https://api.foursquare.com/v2/venues/"
apiSurfixReview_FourSquare = "/tips?sort=recent&limit=500&client_id=G3RGK41B5M5QZWCPY23IQKXZDYK3LHT4TA2HDTAGFL4NFZTD&client_secret=4BF4J3JGWWBKLMEEWVHVZ2P1VFQGJXTBBLRZU0FVSMH4TAHM"

def downloadEarningCalendar(u):
    

        
if __name__ == '__main__':
    bizNames = ["Dominos Pizza", "Domino's Pizza", "Burger King", "Jack in the Box", "McDonald's", "Wendy's", "Wendys", "SONIC Drive In", "Denny's", "IHOP", "Chipotle Mexican Grill"
                , "Panera Bread", "Church's Chicken", "Applebee's", "Jamba Juice", "Chili's Grill & Bar", "Buffalo Wild Wings", "Papa Murphy's", "Olive Garden", "Outback Steakhouse"
                , "El Pollo Loco", "Ruby Tuesday"]
    bizNames = ["Wendys", "SONIC Drive In", "Denny's", "IHOP", "Chipotle Mexican Grill"
                , "Panera Bread", "Church's Chicken", "Applebee's", "Jamba Juice"]
    url = "http://biz.yahoo.com/research/earncal/20150508.html"
    downloadEarningCalendar(url)