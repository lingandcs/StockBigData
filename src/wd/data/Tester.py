'''
Created on Apr 26, 2015

@author: lingandcs
'''
import matplotlib.pyplot as plt
import uuid
import hashlib


if __name__ == '__main__':
    print max([1,4,9,16])
    plt.plot(range(4), [1,4,9,16])
#     plt.axis([0, 6, 0, 20])
#     plt.show()
    print hashlib.md5("whatever your string is").hexdigest()