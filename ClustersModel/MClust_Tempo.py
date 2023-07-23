"""
===========
Clusters
===========
"""

import math
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import fsolve
import Calculador_de_NP as NPCalc
import os
import json
import pandas as pd 
import numpy as np


# Temperatura de calibração
add_temperature = 0
AM = 'AM1'


#Valores para o material
Hm_CTE =    0;              # J/mol -> Entalpia de fusão
D_CTE  =    0;              # m -> aresta do cubo # a = 5.8 b = 14.66 c = 4.806 -> 7.42
Sigma_CTE = 0;              # J/m² -> energia de superfície - Interface cristal - líquido super resfriado
Cs_CTE =    0;              # Calcular taxa a partir de primeiros princípios
P0 =        0;

#Valores para o material
Ns_CTE =    0;              # m² -> Número de sítios por unidade de superfície" # Valor chutado (para forma de pó)
Q_CTE  =    0;              # K/m -> taxa aquecimento

#Temperaturas em Kelvin
Tm_CTE =    0;              # Kelvin Temp de fusão
Tg_CTE =    0;              # Kelvin TemP transição vítrea
Step_CTE =  0;              # Kelvin Passos

#Frenkel
Gama_CTE =  0;              # J/m² - Energia superficial - Interface líquido super resfriado - ar
Ks_CTE =    0;              # Fator de forma
NP_CTE =    0;              # Número de pescoços - fraçoes de 6

#Parâmetros empíricos equação VFT
A_CTE =     0;           
B_CTE =     0;                  # Kelvin
To_CTE =    0;              # Kelvin

#Constantes
R_CTE =     8.3144621;      # J/mol.K -> cte dos gases
Kb_CTE =    1.3806488e-23;  # m².kg/s²K -> boltsmann
Nav_CTE =   6.022e23;       # mol^-1 -> Avogadro
Pi_CTE =    math.pi;

#Limpa a tela do console e deleta as variáveis não utilizadas
print(chr(27) + "[2J")

#Nome da Amostra - Seguir o padrão definido -> Pasta com nome da pessoa e arquivo com o nome da amostra
AMOSTRA_NAME = 'LS2';

#Número mínimo de valores para começar a calcular o Np
MinValuesToCalculateC = 10

"""
===========
BLOCO DE IMPORTAÇÃO DOS ARQUIVOS E CRIAÇÃO DE DIRETÓRIO
===========
"""

#Faz a leitura dos tamanhos de partícula
with open('IMPORT/' + AMOSTRA_NAME + '_PARTICLE_SIZE_' + AM + '.txt') as f:
    Lines =             f.readlines()
    Raio_Import =      [line.split()[0] for line in Lines]
    Fracao_Import =    [line.split()[1] for line in Lines]

#Por padrão, o Python lê como string, converte para ponto flutuante
#Também checa se existe material com o tamanho de partícula descrito antes de adicionar a list
Raio = []; Fracao = []; Number_of_sizes = 0;
for raio, fracao in zip(Raio_Import, Fracao_Import):
    fracao = float(fracao)
    if(fracao != 0):
        Number_of_sizes += 1
        Raio.append(float(raio)/2)
        Fracao.append(float(fracao))

#Faz a leitura dos parâmetros
if os.path.isfile('IMPORT/' + AMOSTRA_NAME + '_PARAMETERS.json'):
    with open('IMPORT/' + AMOSTRA_NAME + '_PARAMETERS.json', "r") as f:
        my_dict = json.load(f)
    locals().update(my_dict)
else:
    print("Não foi possível carregar as variáveis, o software será fechado")
    os._exit(1)


#Cria o diretório de exportação e limpa a variavel Lines (já foi utilizada)
if not os.path.exists("Export/"):
    os.mkdir("Export/")
del Lines

"""
===========
CALCULO DE VARIÁVEIS DEPENDENTES DAS VARIÁVEIS IMPORTADAS
===========
"""

DT_CTE = Tm_CTE - Tg_CTE; 
Vm_CTE = D_CTE**3 * Nav_CTE;    "volume molar do cristal"

"""
===========
BLOCO DE FUNÇÕES (TIPO LAMBDA)
===========
"""

#Eq Ty - Temperatura no tempo y
funcTempY =             lambda y: Tg_CTE + (y * (DT_CTE/((DT_CTE/Step_CTE)-1)))

#Eq VFT - Viscosidade - Pa.s
funcVFT =               lambda Ty: pow(10, A_CTE + (B_CTE/(Ty-To_CTE)))

#Eq Variação energia livre
funcGibbs =             lambda Ty: Hm_CTE * (Tm_CTE - Ty) * Ty / (Tm_CTE**2)

#Eq Sitios pref de crescimento
funcSitiosPref =        lambda dGibbs: Cs_CTE*D_CTE*dGibbs / (4 * Pi_CTE * Sigma_CTE * Vm_CTE)

#Eq difusão efetiva
funcDifusaoEfetiva =    lambda Ty, ny: Kb_CTE * Ty / (D_CTE * ny)

#Eq taxa de crescimento de cristais
funcCrescCrist =        lambda zy, fy, dGibbs, Ty: (zy/D_CTE)*fy*(1-math.exp((-dGibbs/(R_CTE*Ty))))

#Eq fração superficial cristalizada
funcFracaoCrist =       lambda T: (Kb_CTE*T*Cs_CTE*Hm_CTE*(Tm_CTE-T)*T*(1-math.exp(-Hm_CTE*(Tm_CTE-T)/(Tm_CTE**2*R_CTE)))/(D_CTE*Tm_CTE**2*4*Pi_CTE*Sigma_CTE*Vm_CTE*pow(10,(A_CTE+(B_CTE/(T-To_CTE))))))
funcAlfaSf =            lambda FracCrist: (1 - math.exp((-(Pi_CTE*Ns_CTE)*(FracCrist**2)/(Q_CTE**2))))

#Eq modelo de Frenkel
funcpFInt =             lambda T, alfaSf: ((1-alfaSf)/pow(10,(A_CTE+((B_CTE)/(T-To_CTE)))))
funcpF =                lambda raio, pFInt, Ks: (P0/(1-((3*Gama_CTE*Ks*NP_CTE*(pFInt)/(8*raio*Q_CTE)))) **3)

#Eq modelo de Mackenzie-Shuttleworth
funcpMSInt =            lambda T, alfaSf: ((1-alfaSf)/pow(10,(A_CTE+((B_CTE)/(T-To_CTE)))))
funcpMS =               lambda a0, pMSInt: 1-(1-P0)*math.exp((-((3*Gama_CTE*(pMSInt)/(2*a0*Q_CTE)))))

#Eq modelo de Frenkel Corrigido
pFIntC =                lambda T, alfaSf: ((1-alfaSf)/pow(10,(A_CTE+((B_CTE)/(T-To_CTE)))))
pFC =                   lambda raio, pFIntValue: (P0/(1-((3*Gama_CTE*Ks_CTE*NP_CTE*(pFIntValue)/(8*raio*Q_CTE)))))

#Eq modelo de Mackenzie-Shuttleworth Corrigido
pMSIntC =               lambda T, alfaSf: ((1-alfaSf)/pow(10,(A_CTE+((B_CTE)/(T-To_CTE)))))
pMSC =                  lambda raio, pFIntValue: 1*(1-P0)*math.exp((-((3*Gama_CTE*Ks_CTE*NP_CTE*(pFIntValue)/(8*raio*Q_CTE)))))

#Eq de A0
funcA0Int =             lambda T, alfaSf: ((1-alfaSf)/pow(10,(A_CTE+((B_CTE)/(T-To_CTE)))))
funcA0 =                lambda a0, a0Int: (1-(1-P0)*math.exp((-((3*Gama_CTE)/(2*a0*Q_CTE))*a0Int))-0.8)


"""
===========
BLOCO DE FUNÇÕES TRADICIONAIS
===========
"""
# Calcula a temperatura de transição de Frenkel para MS
def calculaT08(pF, jRaio):
    i = 0
    for T in temp_y:
        if(pF[i][jRaio] > 0.8):
            return T
        i += 1
    
# Calcula o A0 para MS
def getA0(r, t08, alfaSf):
    a0 = r/2
    if t08 == None:
        t08 = Tm_CTE - 1
    
    Temp_t = list(range(Tg_CTE, t08+1, Step_CTE))
    A0IntFunc = list(range(Tg_CTE, t08+1, Step_CTE))
    AlfaSf_tA0 = list(range(Tg_CTE, t08+1, Step_CTE))
    for T in Temp_t:
        i = int((T - Tg_CTE)/Step_CTE)
        AlfaSf_tA0[i] = alfaSf[i]
        A0IntFunc[i] = funcA0Int(T, AlfaSf_tA0[i])
        A0Int_t = A0IntFunc[0:i+1]
    a0Int = integrate.simps(A0Int_t, Temp_t, dx=1, axis=-1, even='avg')
    return float(fsolve(funcA0, a0, args=a0Int))

# Função de plotagem das imagens - 2 parâmetros opcionais apenas para quando for plotar 2 curvas juntas
def plotExport(n, xPlot, yPlot, typePlot, xLeg, yLeg, titulo, show, secondXPlot=0, secondYPlot=0):
    print("Salvo:   " + str(n) + " - " + titulo + ".png")
    fig, ax = plt.subplots()
    if(typePlot == "Linear"):
        ax.plot(xPlot, yPlot)
        # Checa se há uma segunda curva a ser plotada e se ela compartilha o eixo Y
        if (secondXPlot != 0):
            if (secondYPlot != 0):
                ax.plot(secondXPlot, secondYPlot)
            else:
                ax.plot(secondXPlot, yPlot)
    elif(typePlot == "LogX"):
        ax.semilogx(xPlot, yPlot)
        # Checa se há uma segunda curva a ser plotada e se ela compartilha o eixo Y
        if secondXPlot != 0:
            if (secondYPlot != 0):
                ax.secondXPlot(secondXPlot, secondYPlot)
            else:
                ax.secondXPlot(secondXPlot, yPlot)
    else:
        ax.semilogy(xPlot, yPlot)
        # Checa se há uma segunda curva a ser plotada e se ela compartilha o eixo Y
        if secondXPlot != 0:
            if (secondYPlot != 0):
                ax.secondXPlot(secondXPlot, secondYPlot)
            else:
                ax.secondXPlot(secondXPlot, yPlot)
    
    ax.set(xlabel=xLeg, ylabel=yLeg, title=titulo)
    ax.grid()
    
    plt.gcf().set_size_inches(8, 5)
    fig.savefig("Export/" + str(n) + " - " + titulo + ".png", dpi = 300, bbox_inches ='tight')
    if show:
        plt.show()
    plt.close(fig)


"""
===========
PROGRAMA PRINCIPAL
===========
"""
Q_CTE_list = [10,100,300,500]
DS_list = ['DS02', 'DS03', 'DS10', 'DS11', 'DS13', 'DS14', 'DS15', 'DS16', 'S7030_3', 'S7030_4', 'S7030_5', 'S7030_6', 'S7030_7', 'S7030_8', 'S7030_9', 'S7030_10', 'N01F12P11', 'N02F12P11']

Q_CTE_list = [10]
with open('Import/LS2_EXP_RATE.json', "r") as f:
    my_dict = json.load(f)
locals().update(my_dict)

temp_q_A = 10
temp_q_B = 10
temp_q_C = 10

#for DS in ['DS15']:
for DS in DS_list:
    Todos_os_tempos = []
    Todos_os_pAmostra = []
    Q_list = []
    
    exec("temp_q_A = "+ DS + "_A")
    exec("temp_q_B = "+ DS + "_B")
    exec("temp_q_C = "+ DS + "_C")
    Q_CTE_list = [temp_q_A,temp_q_B,temp_q_C]
        
    #Faz a leitura da densidade
    if os.path.isfile('IMPORT/Exp0821/' + DS + '.csv'):
        print("Resultado experimental localizado e importado")
        with open('IMPORT/Exp0821/' + DS + '.csv') as f:
            df = pd.read_csv('IMPORT/Exp0821/' + DS + '.csv')
    
        df['pExper'] = df['RETRACAO_MA5'] * P0
    
    else:
        print("Resultado experimental não localizado, a figura 10 não será exportada")
        print("IMPORTANTE: A densidade inicial será a densidade do Json")
        
    Q_CTE = 1
    for Q_CTE in Q_CTE_list:
        #while Q_CTE < 101:
        #Q_CTE = Q_CTE * 10**0.5
        Q_list.append(Q_CTE)
        # O valor de n indica o número de elementos calculados
        n = int(DT_CTE/Step_CTE)
        
        #Calcula uma lista de temperatura no tempo
        temp_y = list(range(Tg_CTE, Tm_CTE, Step_CTE))
        tempo  = [((i-Tg_CTE)/Q_CTE)*60 for i in list(range(Tg_CTE, Tm_CTE, Step_CTE))]
        
        densidade = []
        
        #Define as listas da parte 1
        ty = [];    ny = [];    dGibbs = [];
        fy = [];    zy = [];    uy = [];
        
        #Define as listas da parte 2
        fracaoCrist = [];   alfaSf = [];    
        pFIntFunc = [];     pFInt = [];         pF = [];              
        pMSIntFunc = [];    pMSInt = [];        pMS = [];
        t08 = [];           a0 = [];            pTotal = [];
        pAmostra = [];      dl = [];
        pAmostraMS = [];    pAmostraFr = [];
        
        # Calcula valores em função da TEMPERATURA
        for T in temp_y:
            # Calcula a viscosidade no tempo y
            ny_step = funcVFT(T)
            ny.append(ny_step)
            
            # Calcula a variação de energia livre de Gibbs no tempo y
            dGibbs_step = funcGibbs(T)
            dGibbs.append(dGibbs_step)
            
            # Calcula o número de sítios preferenciais de crescimento
            fy_step = funcSitiosPref(dGibbs_step)
            fy.append(fy_step)
            
            # Calcula a difusão efetiva
            zy_step = funcDifusaoEfetiva(T, ny_step)
            zy.append(zy_step)
            
            # Calcula a taxa de crescimento de cristais
            uy_step = funcCrescCrist(zy_step, fy_step, dGibbs_step, T)
            uy.append(uy_step)
        
        # Calculo de cristalização
        for T in temp_y:
            # Calcula a fração cristalizada
            FracaoC, error = integrate.quad(funcFracaoCrist, Tg_CTE, T)
            fracaoCrist.append(FracaoC)
            
            alfaSf.append(funcAlfaSf(FracaoC))
            
        del FracaoC
        del error
        
        npList = NPCalc.NP_Calc(Raio, Fracao, MinValuesToCalculateC, AMOSTRA_NAME)
        
        # Calcula a densidade de Frenkel
        i = 0
        for T in temp_y:
            pFIntFunc.append(funcpFInt(T, alfaSf[i]))
            pFInt.append(integrate.simps(pFIntFunc[0:i+1], temp_y[0:i+1], dx=1, axis=-1, even='avg'))
            pFLine = []
            for jRaio in range(0, Number_of_sizes):
                density = funcpF(Raio[jRaio],pFInt[i], Ks_CTE)
                if(density < P0 or density > 1):
                    density = 1
                pFLine.append(density)
            pF.append(pFLine)
            i += 1
        del pFIntFunc
        del pFInt
        del pFLine
        
        # Calcula a temperatura que resulta em 80% de densidade e o a0 para MS.
        for jRaio in range(0, Number_of_sizes):
            t08.append(calculaT08(pF, jRaio))
            a0.append(getA0(Raio[jRaio], t08[jRaio], alfaSf))
            
        # Calcula a densidade de Mackenzie-Shuttleworth
        i = 0
        pMSLine2 = []
        pMS2 = []
        for T in temp_y:
            pMSIntFunc.append(funcpMSInt(T, alfaSf[i]))
            pMSInt.append(integrate.simps(pMSIntFunc[0:i+1], temp_y[0:i+1], dx=1, axis=-1, even='avg'))
            pMSLine = []
            for jRaio in range(0, Number_of_sizes):
                density = funcpMS(a0[jRaio], pMSInt[i])  
                if(density < P0 or density > 1):
                    density = 1
                pMSLine.append(density)
                pMSLine2.append(funcpMS(a0[jRaio], pMSInt[i]))
            pMS.append(pMSLine)
            pMS2.append(pMSLine2)
            i += 1
        del pMSIntFunc
        del pMSInt
        del pMSLine
        
        # Calcula a densidade final, combinando Frenkel e Mackenzie-Shuttleworth
        i = 0
        for T in temp_y:
            pTotalLine = []
            somatoria = 0
            for jRaio in range(0, Number_of_sizes):
                if t08[jRaio] == None:
                    t08[jRaio] = Tm_CTE
                if(T < t08[jRaio]):
                    pTotalLine.append(pF[i][jRaio])
                    somatoria += pF[i][jRaio] * Fracao[jRaio]
                else:
                    pTotalLine.append(pMS[i][jRaio])
                    somatoria += pMS[i][jRaio] * Fracao[jRaio]
            pTotalLine.append(somatoria)  
            i += 1
            pTotal.append(pTotalLine)
        del pTotalLine
        
        i = 0
        for T in temp_y:
            somatoria = 0
            somatoriaF = 0
            somatoriaMS = 0
            for jRaio in range(0, Number_of_sizes):
                somatoriaF +=   pF[i][jRaio] * Fracao[jRaio]
                somatoriaMS +=  pMS[i][jRaio] * Fracao[jRaio]
                somatoria += pTotal[i][jRaio] * Fracao[jRaio]
            pAmostraFr.append(somatoriaF)
            pAmostraMS.append(somatoriaMS)
            pAmostra.append(somatoria)
            dl.append(1 - pow((P0/pAmostra[i]), 1/3))
            i += 1
        del somatoria
        del jRaio
        
        # Converte temperatura para Celsius
        converte = False
        temp_celsius = []
        for t in temp_y:
            temp_celsius.append(t - 273)
        if converte:
            temp_y = temp_celsius
            
        temp_celsius = []
        for t in df['TEMPERATURE_MA5']:
            temp_celsius.append(t - 273)
        if converte:
            df['TEMPERATURE_MA5'] = temp_celsius
        df['TEMPERATURE_MA5'] = [x + y for x, y in zip(df['TEMPERATURE_MA5'],[add_temperature]*len(df['TEMPERATURE_MA5']))]
            
        #plotExport(1, temp_y,   ny,         "Log",      "Temperatura (K)",  "Viscosidade",          "Viscosidade vs Temperatura", 0)
        #plotExport(2, temp_y,   uy,         "Log",      "Temperatura (K)",  "Cresc. Cristais",      "Taxa de Crescimento de Cristais vs Temperatura", 0)
        #plotExport(3, temp_y,   alfaSf,     "Linear",   "Temperatura (K)",  "Fração Cristalizada",  "Fração Cristalizada vs Temperatura", 0)
        #plotExport(4, Raio,     Fracao,     "LogX",     "Raio",              "Fração Volumétrica",   "Distribução granulométrica", 0)
        #plotExport(5, temp_y,   pF,         "Linear",   "Temperatura (K)",  "ρF",                   "ρ Frenkel - Densidade relativa vs Temperatura", 0)
        #plotExport(5, temp_y,   pAmostraFr, "Linear",   "Temperatura (K)",  "ρF",                   "ρ Amostra Frenkel - Densidade relativa vs Temperatura", 0)
        #plotExport(6, temp_y,   pMS,        "Linear",   "Temperatura (K)",  "ρMS",                  "ρ Mackenzie-Shuttleworth - Densidade relativa vs Temperatura", 0)
        #plotExport(6, temp_y,   pAmostraMS, "Linear",   "Temperatura (K)",  "ρMS",                  "ρ Amostra Mackenzie-Shuttleworth - Densidade relativa vs Temperatura", 0)
        #plotExport(7, temp_y,   pTotal,     "Linear",   "Temperatura (K)",  "ρC",                   "ρ Compacto - Densidade relativa vs Temperatura", 0)
        #plotExport(8, temp_y,   dl,         "Linear",   "Temperatura (K)",  "Retração",             "Retração vs Temperatura", 0)
        #plotExport(9, temp_y,   pAmostra,   "Linear",   "Temperatura (K)",  "ρAmostra",             "ρ Amostra - Densidade relativa vs Temperatura", 0)
        
        #if os.path.isfile('IMPORT/' + AMOSTRA_NAME + '_EXPERIMENTAL_DENSTITY_' + DS + '.txt'):
        #    plotExport(10,temp_y,   pAmostra,   "Linear",   "Temperatura ºC",  "ρAmostra",             "Compararivo de ρ - Calculado e Experimental", 1, df['TEMPERATURE_MA5'], df['pExper'])
        
        # Comparando 3 modelos e experimental
        fig, ax = plt.subplots()
        #ax.plot(temp_y, pAmostraFr, label="Frenkel Puro")
        #ax.plot(temp_y, pAmostraMS, label="MS Puro")
        ax.plot(tempo, pAmostra, label="Clusters")
        ax.plot(df['TIME_MA5'], df['pExper'], label="Experimental")
        ax.set(xlabel="Tempo", ylabel="ρAmostra", title=f"ρ para {Q_CTE} ºC/min")
        ax.set_xscale('log')
        ax.grid()
        ax.legend(loc="upper left", prop={'size': 8})
        plt.gcf().set_size_inches(8, 5)
        fig.savefig("Export/" + DS + "_Taxa_" + str(Q_CTE) + ".png", dpi = 300)
        #plt.show()
        plt.close(fig)
        Todos_os_tempos.append(tempo)
        Todos_os_pAmostra.append(pAmostra)
        
    takeClosest = lambda num,collection:min(collection,key=lambda x:abs(x-num))
    
    fig, ax = plt.subplots()
    for (pAm, tempo, Q) in zip(Todos_os_pAmostra, Todos_os_tempos, Q_list):
        closest = takeClosest(0.8,pAm)
        index = pAm.index(closest)
        
        #TODO CHECAR SE É DIVIDIDO MESMO
        tempo[:] = [t - tempo[index] for t in tempo]
        ax.plot(tempo, pAm, label=str(Q))
    
    df['pExper'] = df['pExper'].replace(np.nan, P0)
    tExp =  [t/1000 for t in list(df['TIME_MA5'])]
    pExp = list(df['pExper'])

    
    closest = takeClosest(0.8, pExp)
    index = pExp.index(closest)
    tExp[:] = [t - tExp[index] for t in tExp]
    ax.plot(tExp, pExp, label="Experimental")
    
    lim = 1500
    #ax.set_xscale('log')
    ax.legend(loc="upper left", prop={'size': 8})
    plt.gcf().set_size_inches(16, 9)
    plt.xlim([-lim, lim])
    fig.savefig("Export/_" + DS + ".png", dpi = 300)
    #plt.show()
    plt.close(fig)