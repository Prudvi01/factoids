# -*- coding: utf-8 -*-
import gensim
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
from sentence_transformers import SentenceTransformer
import nltk
from test import num_of_revi
from test import clean
#from WikiExtractor import Extractor
nltk.download('punkt')


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stop_words = set(stopwords.words('english'))

#Loading BERT model
#model = SentenceTransformer('bert-base-nli-mean-tokens')
model = SentenceTransformer('albert-base-v2')
'''
#model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M-subword.vec')
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
embed = hub.Module(module_url)

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
#print('Model Loaded')


print("model loaded")

def get_features(texts):
    if type(texts) is str:
        texts = [texts]
    return session.run(embedded_text, feed_dict={text_input: texts})

'''
def get_features(texts):
    if type(texts) is str:
        texts = [texts]
    embedded_text = model.encode(texts)
    return embedded_text

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


def getDistance(articleName):
    numOfRevi = num_of_revi('wiki_data/' + articleName + '.xml')
    revi = 0
    t1 = time.time()
    tree = ec.parse('wiki_data/' + articleName + '.xml')
    result = []
    root = tree.getroot()
    root = root[1]
    Text = ''
    reverts = {}
    bad_chars = [';', ':', '!','*','/','"','\n',',']
    for child in root:
        if 'revision' in child.tag:
            revi += 1
            print('REVISION = ' + str(revi) + '/' + str(numOfRevi))
            for each in child:           
                if 'text' in each.tag:
                    clean_Text = []
                    Text = each.text
                    distance_list = []
                    if(Text!=None):  
                                             
                        Text = knolAnalysis.getCleanText(Text)
                        #Text = clean(Text)
                        print(Text)
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
                                    '''
                                    with open(openFile,'a') as myFile:
                                        myFile.write(str(distance)+' ')
                                    '''
                                    print(clean_Text[-1], clean_Text[-2])
                                    if(distance!=np.inf):
                                        distance_list.append(distance)
                        #if(len(distance_list)!=0):
                        '''
                        with open(openFile,'a') as myFile:
                            myFile.write('\n')
                        '''

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
            
            if revi >= numOfRevi:
                print('elif revi = ' + str(revi))
                break
        

                
                

    t2 = time.time()
    print(t2-t1)
    return result
    #print(clean_Text)


def test(x, a, b, c):
    return a * np.exp(-b * x) + c


def findDistance(article_name):
    distance = getDistance(article_name)

    #print(distance)
    distance = np.array(distance)
    xAxis = [i for i in range(1,len(distance)+1)]
    xAxis = np.array(xAxis)
    '''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        p30 = np.poly1d(np.polyfit(xAxis, distance, 30))  
    '''    
    z = np.polyfit(xAxis, distance, 3)
    p = np.poly1d(z)
    
    xp = np.linspace(0, len(distance), 100)
    #param, param_cov = curve_fit(test, xAxis, distance, bounds=(0, [3., 1., 0.5]))
    
    #print("Sine funcion coefficients:") 
    #print(param) 
    #print("Covariance of coefficients:") 
    #print(param_cov)
          
    '''Below 4 lines can be un-commented for plotting results  
    using matplotlib as shown in the first example. '''
    plt.plot(xAxis, distance, '.', xp, p(xp), '-', lw=1.8)  
    #plt.plot(xAxis, distance, 'b-', lw=1.5,color ='red', label ="data") 
    #plt.plot(xAxis, test(xAxis,*param), '--',lw=1.5, color ='blue', label ="optimized data") 
    #plt.plot(xAxis, p(distance), '--',lw=1.5, color ='blue', label ="optimized data")
    #plt.legend()
    plt.savefig('images/'+article_name+'.png',bbox_inches = "tight",dpi=800)
    plt.show()     
    
    '''
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    #plt.errorbar(xAxis, distance[0], yerr=distance[1], fmt='o', markersize=2, ls='--', lw=0.8, color='black', ecolor='lightgray', elinewidth=0.7, capsize=0)
    ax.set_xlabel('Revisions')
    
    ax.set_ylabel('Average of Distance')
    
    ax.plot(xAxis,distance,color=col,linewidth=2.0)
    
    plt.show()                    
    '''


'''
fileNames = os.listdir('results/')
for f in fileNames:
    if not f == '.DS_Store':
        print('1')
        findDistance(f[:-4])
        print('')
        print("Article "+f[:-4]+" is done:")
'''

findDistance('Bombing_of_Singapore_(1944–45)')