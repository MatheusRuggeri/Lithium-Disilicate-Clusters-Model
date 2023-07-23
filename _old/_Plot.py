# -*- coding: utf-8 -*-
"""
PROBLEMA ENCONTRADO:
    Diversos valores de Y para um mesmo X, pois o eixo Y é calculado a cada 16 ms, o X, 200 ms
    Esse software junta valores iguais em um só e tira a média.

"""

import os, csv
import matplotlib.pyplot as plt
import pandas as pd 

DS_list = ['DS02', 'DS03', 'DS10', 'DS11', 'DS13', 'DS14', 'S7030_3', 'S7030_4', 'S7030_5', 'S7030_6', 'S7030_7', 'S7030_8', 'S7030_9', 'S7030_10', 'N01F12P11', 'N02F12P11']

for DS in DS_list:
    print(DS)
    #df = pd.read_csv(DS + "/Distances_" + DS + ".csv", header=None)
    df = pd.read_csv(DS + "/out.csv")

    fig = plt.figure(figsize=(16,9))
    plt.title(DS + " - Temperatura x Tamanho")
    plt.plot(df['TEMPERATURE'], df['DISTANCE'])
    plt.savefig("Z" + DS + '_Temperatura.jpg', dpi = 120)
    
    fig = plt.figure(figsize=(16,9))
    plt.title(DS + " - Tempo x Tamanho")
    plt.plot(df['TIME'], df['DISTANCE'])
    plt.savefig("Z" + DS + '_Tempo.jpg', dpi = 120)
    
    
    df.to_csv(DS + '.csv', index=False) 