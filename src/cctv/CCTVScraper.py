'''
Created on Jul 11, 2015

@author: lingandcs
'''

from bs4 import BeautifulSoup
import requests
from datetime import datetime
import csv
from dateutil.parser import parse
from datetime import timedelta
import sys


SESSION = requests.Session()

def getWebContent(u):
#     response = br.open(u)
    
    r = SESSION.get(u)
    soup = BeautifulSoup(r.content)
    newsList = soup.find_all(class_ = 'title2 fs_14')
    for item in newsList:
        for news in item.find_all('li'):
           # print news.string+'\n'
            r = SESSION.get(news.a.get('href'))
            soup = BeautifulSoup(r.content)
            text = soup.find(class_= 'body')
            print news.string.encode('gbk')
            
#                 f.write('<b>'+news.string.encode('gb2312')+'</b>'+'\n')
#                 f.write(text.encode('gb2312')+'\n')
    
    
#return: list of new info: [headline, body]
def getNewsTitle(u):
    r = SESSION.get(u)
    webText = r.content
    soup = BeautifulSoup(webText)
    newsSections = soup.find_all(class_ = 'title2 fs_14') 
    allNews = []
    for section in newsSections:
        newsTitles = section.find_all("li")
#         print newsTitles
        for newsTitle in newsTitles:
            headline = newsTitle.get_text().encode('utf8')
#             print headline
            urlEachNews = newsTitle.a.get("href")
            newsPage = BeautifulSoup(SESSION.get(urlEachNews).content)
#             print parseNews
            newsText = newsPage.find("div", attrs = {'class':'body'})
            newsDiv = newsPage.find('div', attrs = {'id':'content_body'})
            ps = None
            try:
                ps = newsDiv.findAll('p')
            except AttributeError:
                print 'no p tag'
            newsBody = []
            if ps != None:
                for p in ps:
                    if p == None or p.string == None:
                        continue
                    
                    paragraph = p.get_text().encode('utf8')
#                     print paragraph
                    newsBody.append(paragraph)
#                     if p.findAll('strong') != None:
#                         print p.text
                    if p.find('strong') != None:
#                         print '<TITLE>' + p.string.strip() + '</TITLE>'#guonei lianbo kuaixun
                        pass
                    else:
                        pass
#                         print p.string.strip()
#                     print '-----------'
            newsBody = '\n'.join(newsBody)
            allNews.append([headline, newsBody])
    return allNews

def downloadXWLB(startDTStr, endDTStr, outputPath):
    
    if isinstance(startDTStr, basestring) == True:
        startDT = parse(startDTStr)
    else:
        startDT = startDTStr
        
    if isinstance(endDTStr, basestring) == True:
        endDT = parse(endDTStr)
    else:
        endDT = endDTStr
        
    d = startDT
    delta = timedelta(days=1)
        
    handle = open(outputPath + startDT.strftime('%Y%m%d') + ".csv", "wb")
    csvWriter = csv.writer(handle)
    while d <= endDT:
        print 'processing', d.strftime('%Y%m%d')
        u = 'http://cctv.cntv.cn/lm/xinwenlianbo/' + d.strftime('%Y%m%d') + '.shtml'
#         print u
        allNews = getNewsTitle(u)
        for news in allNews:
            csvWriter.writerow([d.strftime('%Y%m%d'), news[0], news[1]])
        d += delta
    handle.close()
    
#detect the date when work ended last time
def detectPreviousDate(path):
    handle = open(path, "rb")
    csvReader = csv.reader(handle)
    latestDT = None
    for row in csvReader:
        dt = row[0]
#         print dt
        if latestDT == None or parse(dt) > latestDT:
            latestDT = parse(dt)
    handle.close()
    return latestDT
    
    
if __name__ == '__main__':
    outputFolder = "D:\\projects\\StockSentiment\\data\\"
#     outputLast = "D:\\projects\\StockSentiment\\data\\xwlb.csv"  
#     previousDT = detectPreviousDate(outputLast)
#     print 'previosu end date:', previousDT
#     if previousDT == None:
#         previousDT = '20150701'
    
    
    startDate = '20140704'
    endDate = '20141231'
    downloadXWLB(startDate, endDate, outputFolder)    
    