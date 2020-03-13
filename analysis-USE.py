# -*- coding: utf-8 -*-
import xml.etree.cElementTree as ec
import mwparserfromhell
import matplotlib.pyplot as plt
import numpy as np
from dateutil.parser import parse
from datetime import datetime
from kdap.analysis import knolAnalysis
import nltk.data
import re
import time
import tensorflow as tf
import tensorflow_hub as hub
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from scipy.optimize import curve_fit 
import warnings
import os
import sys
from sentence_transformers import SentenceTransformer
import nltk
from test import num_of_revi
start_time = time.time()

nltk.download('punkt')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stop_words = set(stopwords.words('english'))


#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.compat.v1.disable_eager_execution()

#model=gensim.models.KeyedVectors.load_word2vec_format("wiki-news-300d-1M-subword.vec")
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
embed = hub.Module(module_url)
print("model loaded")
g = tf.Graph()

with g.as_default():
  # We will be feeding 1D tensors of text into the graph.
  text_input = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
  embedded_text = embed(text_input)
  init_op = tf.group([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])

g.finalize()

session = tf.compat.v1.Session(graph=g)
session.run(init_op)

result = []

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s/%s %s%% %s' % (prefix, bar, iteration, total, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
def get_features(texts):
    if type(texts) is str:
        texts = [texts]
    return session.run(embedded_text, feed_dict={text_input: texts})

def cosine_similarity(v1, v2):
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if (not mag1) or (not mag2):
        return 0
    return np.dot(v1, v2) / (mag1 * mag2)

def test_similarity(text1, text2):
    vec1 = get_features(text1)[0]
    vec2 = get_features(text2)[0]
    #print(vec1.shape)
    return cosine_similarity(vec1, vec2)

def getDist(article_name, revilimit):
    numOfRevi = num_of_revi('wiki_data/' + article_name + '.xml')
    revi = 0
    printProgressBar(0, revilimit, prefix = 'Progress:', suffix = 'Complete', length = 50)
    t1 = time.time()
    tree = ec.parse('wiki_data/' + article_name + '.xml')
    result = []
    root = tree.getroot()
    root = root[1]
    Text = ''
    reverts = {}
    bad_chars = [';', ':', '!','*','/','"','\n',',']
    for child in root:
        if 'revision' in child.tag:
            revi += 1
            printProgressBar(revi, revilimit, prefix = article_name, suffix = 'Complete', length = 50)
            #print('REVISION = ' + str(revi) + '/' + str(numOfRevi))
            for each in child:           
                if 'text' in each.tag:
                    clean_Text = []
                    Text = each.text
                    distance_list = []
                    if(Text!=None):  
                                             
                        Text = knolAnalysis.getCleanText(Text)
                        #Text = clean(Text)
                        #print(Text)
                        sent_list = tokenizer.tokenize(Text)
                        sent_num = 0
                        
                        for sen in sent_list:
                            if(not sen.isspace()):
                                sent_num+=1
                                if(sen[-1]=='.'):
                                    sen = sen[:-1]
                                S = re.sub('==[^>]+==', '', sen)
                                S = S.replace('\xa0',' ')
                                S = ''.join(i for i in S if not i in bad_chars)
                                word_tokens = word_tokenize(S)
                                filtered_sentence = [w for w in word_tokens if not w in stop_words]                                
                                clean_Text.append(filtered_sentence)
                                if(sent_num>1):
                                    distance = test_similarity(str(clean_Text[-1]),str(clean_Text[-2]))
                                    
                                    if(distance!=np.inf):
                                        distance_list.append(distance)
                    
                        distance_avg = np.average(distance_list)
                            #standardDev = np.std(distance_list)
                            #if(distance_avg>0.45 and distance_avg<0.64):
                        if(distance_avg!=np.inf and (not np.isnan(distance_avg))):
                            result.append(distance_avg)

                
                if 'sha1' in each.tag:
                    sha1Value = each.text
                    try:
                        if reverts[sha1Value]:
                           result.pop(-2)
                            
                    except:
                        reverts[sha1Value] = 1
            
            if revi >= revilimit:
                print('elif revi = ' + str(revi))
                break
        

                
                

    t2 = time.time()
    print(t2-t1)
    return result
        #print(required)
        #print(len(required))
    
def plotDist(article_name, revilimit):
    result = getDist(article_name, revilimit)
    xAxis = [i for i in range(1,len(result)+1)]
    
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    #plt.errorbar(xAxis, distance[0], yerr=distance[1], fmt='o', markersize=2, ls='--', lw=0.8, color='black', ecolor='lightgray', elinewidth=0.7, capsize=0)
    ax.set_xlabel('Revisions')
    
    ax.set_ylabel('Average of Distance')
    
    ax.plot(xAxis,result,color='coral',linewidth=2.0)
    plt.savefig('images/USE/'+article_name+'USErev_'+str(revilimit)+'.png',bbox_inches = "tight",dpi=800)
    #plt.show()
    print("--- Time taken to execute: %s seconds ---" % (time.time() - start_time))

fileNames = os.listdir('wiki_data/')
for article_name in fileNames:
    if not article_name == '.DS_Store' and not article_name == '.gitignore':
        arguments = sys.argv
        numOfRevi = num_of_revi('wiki_data/' + article_name)
        if len(arguments) < 2:
            revilimit = numOfRevi
        else:
            revilimit = int(sys.argv[1])

        plotDist(article_name[:-4], revilimit)
        print('')
        print("Article "+article_name[:-4]+" is done:")
        
'''
article_name = 'Bombing_of_Singapore_(1944–45)'
arguments = sys.argv
numOfRevi = num_of_revi('wiki_data/' + article_name + '.xml')
if len(arguments) < 2:
    revilimit = numOfRevi
else:
    revilimit = int(sys.argv[1])

plotDist(article_name, revilimit)
'''