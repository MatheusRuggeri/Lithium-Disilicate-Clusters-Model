# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 23:41:45 2021

@author: jomar
"""

from os import listdir
from os.path import isfile, join
import pandas as pd

onlyfiles = [f for f in listdir() if isfile(join("", f))]
onlyexcel = [f for f in onlyfiles if '.xls' in f]

del onlyfiles


for file_name in onlyexcel:
    df = pd.read_excel(file_name, sheet_name='Sheet')
    df = df.drop(columns=['Size Low µm', 'Size High µm','Q3(x) %','100-Q3(x) %','q3(x) 1/mm','q3log(x)'])
    df['Size Mid µm'] = df['Size Mid µm'] * 1E-6
    df['dQ3(x) %'] = df['dQ3(x) %'] / 100
    
    df.to_csv('LS2_PARTICLE_SIZE_' + file_name[:3].upper() + '.txt', index=False, sep='\t', header=False)