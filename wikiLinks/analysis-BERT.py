# Analysis of sentences using BERT
import xml.etree.ElementTree as ET
import numpy as np
#import gensim
#from gensim.models import FastText as f
import time
import mwparserfromhell
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import scipy as sp
from sentence_transformers import SentenceTransformer

#tf.enable_eager_execution()
start_time = time.time()

#Loading model
model = SentenceTransformer('bert-base-nli-mean-tokens')

result = []

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

def getDist(article_name):
    wordPairDict = {}
    tree = ET.parse(article_name)
    root = tree.getroot()
    for terms in root[1].findall('{http://www.mediawiki.org/xml/export-0.10/}revision'):
        text=terms.find('{http://www.mediawiki.org/xml/export-0.10/}text').text
        wikicode = mwparserfromhell.parse(text)
        wikiLinks = wikicode.filter_wikilinks()
        d={}
        for link in wikiLinks:
            if(('File:' not in link) and ('Category:' not in link) and ('Image:' not in link)):
                index=text.find(str(link.title))
                strg=len(str(link.title))
                extra='!'*strg
                text=text[:index]+extra+text[(index+strg):]
                d[index]=str(link.title)
        for link in wikiLinks:
            if(('File:' not in link) and ('Category:' not in link) and ('Image:' not in link)):
                c=text.count(str(link.title))
                l=[]
                while c>0:
                    index=text.find(str(link.title))
                    if index>0:
                        l.append(index)
                    strg=len(str(link.title))
                    extra='!'*strg
                    text=text[:index]+extra+text[(index+strg):]
                    c-=1
                for ind in l:
                    d[ind]=str(link.title)
        required=[]
        for elem in sorted(d):
            required.append(d[elem])
        
        distance_list = []    
        for i in range(len(required)):
            if((i!=len(required)-1) and (required[i]!=None and required[i+1]!=None)):
                dummyPair = str(required[i])+'#$#'+str(required[i+1])
                if(wordPairDict.get(dummyPair)==None):
                    distance = test_similarity(str(required[i]),str(required[i+1]))
                    wordPairDict[dummyPair] = distance
                else:
                    distance = wordPairDict[dummyPair]
                if(distance!=np.inf):
                    distance_list.append(distance)
                    
                    
        if(len(distance_list)!=0):
            distance_avg = np.average(distance_list)
            #standardDev = np.std(distance_list)
            
            result.append(distance_avg)  

    return result
    
def plotDist(article_name):
    result = getDist(article_name)
    xAxis = [i for i in range(1,len(result)+1)]
    
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    #plt.errorbar(xAxis, distance[0], yerr=distance[1], fmt='o', markersize=2, ls='--', lw=0.8, color='black', ecolor='lightgray', elinewidth=0.7, capsize=0)
    ax.set_xlabel('Revisions')
    
    ax.set_ylabel('Average of Distance')
    
    ax.plot(xAxis,result,color='coral',linewidth=2.0)
    
    plt.show()
    print("--- Time taken to execute: %s seconds ---" % (time.time() - start_time))

article_name = '2006_Westchester_County_tornado.xml'
plotDist(article_name)