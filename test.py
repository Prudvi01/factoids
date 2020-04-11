# -*- coding: utf-8 -*-
from helper import num_of_revi
import os
import sys
import time
import xml.etree.cElementTree as ec
from kdap.analysis import knolAnalysis

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

def findsent(dire, article_name, revilimit):
    numOfRevi = num_of_revi(dire + article_name + '.xml')
    revi = 0
    print(article_name)
    printProgressBar(0, revilimit, prefix = 'Progress:', suffix = 'Complete', length = 50)
    t1 = time.time()
    context_wiki = ec.iterparse(dire + article_name + '.xml', events=("start","end"))
    context_wiki = iter(context_wiki)
    result = []
    Text = ''
    reverts = {}
    bad_chars = [';', ':', '!','*','/','"','\n',',']
    event_wiki, root_wiki = next(context_wiki)
    for event, elem in context_wiki:
        if event == "end" and 'revision' in elem.tag:
            revi += 1
            printProgressBar(revi, revilimit, prefix = article_name, suffix = 'Complete', length = 50)
            #print('REVISION = ' + str(revi) + '/' + str(numOfRevi))
            for each in elem:           
                if 'text' in each.tag:
                    clean_Text = []
                    Text = each.text
                    distance_list = []
                    if(Text!=None):  
                                             
                        Text = knolAnalysis.getCleanText(Text)
                        #sent_list = tokenizer.tokenize(Text)
                        result.append(len(Text))
    x = 0
    results = []
    for i in result:
        results.append(i - x)
        x = i

    return results

def run(dire):
    completedfile = open("completedusetest.txt", "r")
    completed = completedfile.readlines()
    completedfile.close()
    fileNames = os.listdir(str(dire))
    for article_name in fileNames:
        if not article_name == '.DS_Store' and not article_name == '.gitignore':
            if not (article_name[:-4] + '\n') in completed:    
                arguments = sys.argv
                numOfRevi = num_of_revi(dire + article_name)
                if len(arguments) < 2:
                    revilimit = numOfRevi
                else:
                    revilimit = int(sys.argv[1])

                result = findsent(dire, article_name[:-4], revilimit)
                # Write the distances to a file
                with open("sentresults/"+article_name+'USErev_'+str(revilimit)+'.txt', 'w') as filehandle:
                    filehandle.writelines("%s\n" % r for r in result)             
                #plotDist(article_name[:-4], revilimit)
                print('')
                f = open("completeduse.txt", "a")
                f.write(article_name[:-4] + '\n')
                f.close()
                print("Article "+article_name[:-4]+" is done:")
                
            else:
                print('Skipping ' + article_name[:-4])

run('data_set/')