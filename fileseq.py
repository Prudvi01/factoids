# -*- coding: utf-8 -*-
import os
from django.utils.encoding import smart_str
dire = 'data_set'
fileNames = (os.listdir(str(dire)))
with open('fileseq.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % smart_str(r) for r in fileNames)