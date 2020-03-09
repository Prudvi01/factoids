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
from test import clean
#from WikiExtractor import Extractor
nltk.download('punkt')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stop_words = set(stopwords.words('english'))

#Loading BERT model
model = SentenceTransformer('bert-base-nli-mean-tokens')

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
    numOfRevi = num_of_revi(sys.argv[1] + articleName + '.xml')
    revi = 0
    t1 = time.time()
    tree = ec.parse(sys.argv[1] + articleName + '.xml')
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
    plt.plot(xAxis, distance, 'b-', lw=1.5,color ='red', label ="data") 
    plt.plot(xAxis, test(xAxis,*param), '--',lw=1.5, color ='blue', label ="optimized data") 
    plt.plot(xAxis, p(distance), '--',lw=1.5, color ='blue', label ="optimized data")
    plt.legend()
    plt.savefig('images/'+article_name+'.png',bbox_inches = "tight",dpi=800)

'''
fileNames = os.listdir('results/')
for f in fileNames:
    if not f == '.DS_Store':
        print('1')
        findDistance(f[:-4])
        print('')
        print("Article "+f[:-4]+" is done:")
'''
arguments = sys.argv
print(arguments)
if len(arguments) < 2:
    print('No file path mentioned. \nUsage "python analysis-bert.py path/to/the/xml/file/in/directory/"')
    exit(1)

findDistance('Bombing_of_Singapore_(1944–45)')