# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:37:05 2020

@author: jomar
"""

import math
import matplotlib.pyplot as plt
import Aproximador_de_curvas as AproxCurve

def NP_Calc(Raio, Fracao, MinValuesToCalculateC, AMOSTRA_NAME, printValue=False):
    "Soma as frações e mostra na tela (deve ficar muito próxima a 100%)"    
    total = 0
    for f in Fracao:
        total += f
    if printValue:
        print("Somatória das frações: ", end="")
        print(round(total*100,2), end="%\n")
    
    npList = []
    lnR = []
    lnSum = []
    nList = []
    
    n = lambda r, rk: ((2 * math.pi)/ math.sqrt(3)) * ((2/3) + ((2*rk)/r) + ((rk/r)**2))
    
    "Calcula o ln da soma e o ln do raio"
    for r in Raio:
        SumK = 0
        for rk, ϕk in zip(Raio, Fracao):
            nList.append(n(r,rk))
            SumK += n(r,rk) * ϕk
        lnSum.append(math.log(SumK))
        
        lnR.append(math.log(r))
    
    c = AproxCurve.calcula_c(lnR, lnSum, MinValuesToCalculateC, AMOSTRA_NAME)
    if printValue:
        print("Valor de C = ", end="")
        print(c)
        
    for r in Raio:
        SumR = 0
        for rr, ϕr in zip(Raio, Fracao):
            SumR += (ϕr/pow(rr,c))
        ξr = 0
        ξr = ((1/pow(r,c)) / SumR)
        if ξr < 1:
            ξr = 1
        npList.append(ξr)
        #print(r, end='\t\t')
        #print(ξr)
        
    #print(c)
    #print(npList)
    #print(lnR)
    #print(lnSum)
    
    # Plota as 2 juntas
    fig, ax = plt.subplots()
    ax.plot(Raio, npList)
    ax.set(xlabel="Raio", ylabel="Np List", title="")
    ax.grid()
    plt.gcf().set_size_inches(8, 5)
    fig.savefig("Export/" + AMOSTRA_NAME + "/0 - NP.png", dpi = 300)
    plt.close(fig)        
    
    # Plota as 2 juntas
    fig, ax = plt.subplots()
    ax.plot(lnR, lnSum)
    ax.set(xlabel="ln(r)", ylabel="ln(Somatória)", title="")
    ax.grid()
    plt.gcf().set_size_inches(8, 5)
    fig.savefig("Export/" + AMOSTRA_NAME + "/0 - Raio-Sum(K).png", dpi = 300)
    plt.close(fig)

    return npList