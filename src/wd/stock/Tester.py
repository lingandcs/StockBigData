'''
Created on Apr 15, 2015

@author: lingandcs
'''
import nltk
import sklearn
import scipy
import numpy
from pylab import *



if __name__ == '__main__':
    print 'hello world!'
    x = linspace(0, 5, 10)
    y = x ** 2
    subplot(1,2,1)
    plot(x, y, 'r--')
    subplot(1,2,2)
    plot(y, x, 'g*-');