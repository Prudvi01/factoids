import os, sys
import matplotlib.pyplot as plt
import numpy as np
from helper import num_of_revi, findPosper, justsent, chunkIt

def phaseAvgs(dire, article_name):
    filelist = open(dire + article_name, 'r')
    filelines = filelist.readlines()
    distances = []
    for tempo in filelines:
        distances.append(float(tempo[:-1]))
    distances = justsent(distances)
    phases = chunkIt(distances, 4)

    stds = (np.std(phases[0]), np.std(phases[1]), np.std(phases[2]), np.std(phases[3]))
    phaseavgs = (np.average(phases[0]), np.average(phases[1]), np.average(phases[2]), np.average(phases[3]))

    return phaseavgs, stds

def run(dire):
    phaseavgfile = open("phaseavgfile.txt", "a")
    fileNames = os.listdir(str(dire))
    for article_name in fileNames:
        if os.path.getsize(dire + article_name) > 0:
            if not article_name == '.DS_Store' and not article_name == '.gitignore':
                phaseavgs, stds = phaseAvgs(dire, article_name)
                phaseavgfile.write(article_name[:-4] + '\n')
                phaseavgfile.write(str(phaseavgs) + '\n')
                phaseavgfile.write(str(stds) + '\n')
                print('')
                print("Article "+article_name[:-4]+" is done:")
    phaseavgfile.close()
run('result_final/')

'''
outputfile format:

Article name
phaseavg
STD
'''