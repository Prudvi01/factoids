# -*- coding: utf-8 -*-
import os
dire = 'data_set'
fileNames = os.listdir(str(dire))
with open('fileseq.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % r for r in fileNames)