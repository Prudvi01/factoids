import os, sys
import matplotlib.pyplot as plt
import numpy as np
from helper import num_of_revi, findPosper
def plotDist(dire, article_name):
    # Open files and get xAxis and yAxis values
    posfile1D = open("positivedegBD1.txt", "a")
    posfile3D = open("positivedegBD3.txt", "a")
    # yAxis
    filelist = open(dire + article_name + '.txt', 'r')
    temp = filelist.readlines()
    yAxis = []
    for tempo in temp:
        yAxis.append(float(tempo[:-1]))
    # xAxis
    resultlength = len(yAxis)
    xAxis = [i for i in range(1,len(yAxis)+1)]
    xAxis = np.array(xAxis) 
    
    # Plotting the data on graph
    #plt.plot(xAxis, result, 'o')
    plt.style.use('fivethirtyeight')
    plt.xlabel('Revisions')
    plt.ylabel('Similarity')
    plt.suptitle(article_name, fontsize = 16)

    # Fitting the data with linear and polynomial lines
    z = np.polyfit(xAxis, yAxis, 1)
    y = np.polyfit(xAxis, yAxis, 3)
    p = np.poly1d(z)
    p3 = np.poly1d(y)
    xp = np.linspace(0, len(yAxis), 100)
    plt.plot(xAxis, yAxis, '.', xp, p(xp), '-', xp, p3(xp), '--', lw=1.8)
    
    # Finding the slope data to find the positive percent
    derivative = np.polyder(p) # Linear
    derivative3 = np.polyder(p3) # Ploynomial Deg 3
    a = [(np.polyval(derivative,i)) for i in xAxis]
    posper = findPosper(a, resultlength)
    posfile1D.write(article_name[:-4] + ' = ' + str(posper) + '% positive.''\n')
    b = [(np.polyval(derivative3,i)) for i in xAxis]
    posper3 = findPosper(b, resultlength)
    posfile3D.write(article_name[:-4] + ' = ' + str(posper3) + '% positive.''\n')
    
    # Save the graph and close the files
    plt.savefig('images/USE1and3/'+article_name+'USE.png',bbox_inches = "tight",dpi=800)
    posfile1D.close()
    posfile3D.close()


def run(dire):
    fileNames = os.listdir(str(dire))
    for article_name in fileNames:
        if os.path.getsize(dire + article_name) > 0:
            if not article_name == '.DS_Store' and not article_name == '.gitignore':
                plotDist(dire, article_name[:-4])
                print('')
                print("Article "+article_name[:-4]+" is done:")
                
run('result_final/')