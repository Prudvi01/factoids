# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

#xAxis = []
#yAxis = []
file_name = os.listdir('result_final')

for f in file_name:
    count = 1
    xAxis = []
    yAxis = []
    with open('result_final/'+f, 'r') as myFile:
        for line in myFile:
            yAxis.append(float(line))
            xAxis.append(count)
            count+=1
    print(yAxis)
    z = np.polyfit(xAxis, yAxis, 1)
    y = np.polyfit(xAxis, yAxis, 3)
    p = np.poly1d(z)
    p3 = np.poly1d(y)
    xp = np.linspace(0, len(yAxis), 100)
    plt.plot(yAxis, '.', xp, p(xp), '-', xp, p3(xp), '--', lw=1.8)
    #plt.scatter(xAxis,yAxis)
    #plt.show()
    plt.savefig('images/USE1and3/'+f+'USE.png',bbox_inches = "tight",dpi=800)
    print('')
    print("Article "+f+" is done:")
