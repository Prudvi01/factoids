import os, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from helper import num_of_revi, findPosper, justsent, chunkIt

def findpearson(dire, article_name):
    filelist = open(dire + article_name, 'r')
    filelines = filelist.readlines()
    distances = []
    for tempo in filelines:
        distances.append(float(tempo.strip()))
    xAxis = np.arange(0, len(distances))

    pearson = stats.pearsonr(xAxis, distances)
    return pearson

def run(dire):
    phaseavgfile = open("pearsonanalysis.txt", "a")
    fileNames = os.listdir(str(dire))
    for article_name in fileNames:
        if os.path.getsize(dire + article_name) > 0:
            if not article_name == '.DS_Store' and not article_name == '.gitignore':
                pearson = findpearson(dire, article_name)
                phaseavgfile.write(article_name[:-4] + '\n')
                phaseavgfile.write(str(pearson) + '\n')
                print('')
                print("Article "+article_name[:-4]+" is done:")
    phaseavgfile.close()
run('result_final/')