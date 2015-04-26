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


apiPrefixReview_FourSquare = "https://api.foursquare.com/v2/venues/"
apiSurfixReview_FourSquare = "/tips?sort=recent&limit=500&client_id=G3RGK41B5M5QZWCPY23IQKXZDYK3LHT4TA2HDTAGFL4NFZTD&client_secret=4BF4J3JGWWBKLMEEWVHVZ2P1VFQGJXTBBLRZU0FVSMH4TAHM"



if __name__ == '__main__':
    conn = mysql.connector.connect(user='lingandcs', password='sduonline',
                              host='107.170.18.102',
                              database='goodfoodDB')
    query_insert = "INSERT INTO goodfoodDB.FinanceNLP_review_FS (id,bizName,createAt,bizSrcID) "  \
                    "VALUES (%s,%s,%s,%s)"

    try:
        cursor = conn.cursor()
        cursor_insert = conn.cursor()
        
        #Hard code biz name, to be changed
        cursor.execute("""
            SELECT * FROM goodfoodDB.goodfood_biz_FourSquare
            WHERE bizName = 'Starbucks'
        """)
        
        result = cursor.fetchall()
        print "%d business loaded!"% len(result)
        bizIDs = [row[3] for row in result]
#         print bizIDs
        today = time.strftime("%Y%m%d")
        countReview = 0
        countBiz = 0
        
        for biz in result:
            countBiz += 1
            if countBiz % 100 == 0:
                print "%dth biz processed"% countBiz
            bizSrcID = biz[3]
            bizName = biz[1] 
            apiURL = apiPrefixReview_FourSquare + bizSrcID + apiSurfixReview_FourSquare + "&v=" + today
#             print apiURL
            try:
                jsonResult = json.load(urllib2.urlopen(apiURL))
#                 print jsonResult['response']['tips']['count']
                reviews = jsonResult['response']['tips']['items']
                for review in reviews:
                    reviewID = review['id']
                    createAt = review['createdAt']
                    canonicalUrl = review['canonicalUrl']
                    createAt = time.strftime('%Y-%m-%d', time.localtime(int(createAt)))
#                     print reviewID, bizName, createAt, bizSrcID
                    try:
                        cursor_insert.execute(query_insert, (reviewID, bizName, createAt, bizSrcID))
                        print cursor.lastrowid
                        countReview += 1
                        if countReview%1000 == 0:
                            print '%d reviews inserted'% countReview
                    except mysql.connector.errors.IntegrityError:
#                         print 'fuck, duplicate key!!!'
                        pass
#                     conn.commit()
#                 print len(reviews)
#                     sys.exit()
            except urllib2.HTTPError:
                print 'fucking network!'

        
    finally:
        conn.commit()
        conn.close()