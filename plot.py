# importing the required module 
import matplotlib.pyplot as plt 
import pyperclip as pc
import numpy as np
import seaborn as sns
import pandas as pd
import os, sys

def justsent(temp, limit = 2000):
    x = 0
    positive = []
    negative = []
    for i in temp:
        newsent = i - x
        if newsent <= limit and newsent >= - limit:
            if newsent <= 0:
                negative.append(newsent)
            if newsent >= 0:
                positive.append(newsent)
            x = i
        else:
            yaxis.append(0)

    return positive, negative

dire = 'data/'
names = ['Alexander the Great ','Avatar (2009 film) ','California ']
limits = [38, 100, 200]
l = 0
for article_name in names:
    article_name = article_name
    filelist = open(dire + article_name + '.txt', 'r')
    results = filelist.readlines()
    temp = []
    yaxis =[]
    #print(results)
    for result in results:
        temp.append(float(result[:-1]))


    positive, negative = justsent(temp, limits[l])
    l += 1
    #plt.style.use('dark_background')
    '''
    for i in yaxis:
        if i >= 0:
            positive.append(i)
        else:
            negative.append(i)
            '''
    # x axis values 
    xAxis = [i for i in range(1,len(positive)+1)]
    xAxis = np.array(xAxis)
    pos=pd.DataFrame({'x': xAxis, 'y': positive})
    # plotting the points  
    plt.plot( 'x', 'y', data=pos, marker='o', markerfacecolor='green', markersize=1, color='purple', linewidth=0.5)

    xAxis = [i for i in range(1,len(negative)+1)]
    xAxis = np.array(xAxis)
    neg=pd.DataFrame({'x': xAxis, 'y': negative})
    plt.plot( 'x', 'y', data=neg, marker='x', markerfacecolor='red', markersize=2, color='red', linewidth=0.5)

    plt.style.use('fivethirtyeight')
    plt.xlabel('Revisions', fontsize = 14)
    plt.ylabel('New sentences', fontsize = 14)
    #plt.title(article_name, fontsize = 16)

    # function to show the plot 
    print(len(negative))

    plt.savefig('sentresults/' + article_name + '.png',bbox_inches = "tight",dpi=800, facecolor = "#ffffff")