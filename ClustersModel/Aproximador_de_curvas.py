# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:04:17 2020

@author: jomar
"""


import numpy as np
import matplotlib.pyplot as plt

MIN_VALUES = 10

#lnR = [-17.167917775457, -17.072607595652673, -16.985596218663044, -16.8838135243531, -16.791440204222084, -16.69791414621126, -16.6123919727731, -16.518573217555446, -16.41920074374224, -16.328816682273974, -16.234629467214273, -16.14341345894261, -16.050437002484504, -15.956827503362197, -15.86345343258474, -15.76743877934515, -15.676620105327123, -15.581602280443752, -15.489486991535946, -15.395389668156831, -15.302730837674126, -15.209837090781429, -15.115627222843113, -15.022822263555724, -14.930252228562267, -14.835773880221408, -14.743115049738703, -14.650221302846006, -14.556798113852155, -14.459867574354787, -14.368895796149062, -14.277546017560832, -14.17935399138162, -14.089947403666034, -13.995834112095556, -13.90434177167089, -13.810523016453235, -13.715665222994557, -13.624890198355624, -13.530331615730612, -13.437074122244029, -13.342386801382295, -13.250196748914213, -13.15799055504748, -13.064094469280352, -12.969642290386664, -12.877458334308606, -12.784107018989614, -12.69058096097879, -12.597634848469347, -12.50447868134493, -12.410639811271448, -12.318122149338796, -12.224236616157844, -12.129111604394046, -12.0405582070526, -11.943708381062683, -11.848398201258357, -11.754997026169956, -11.663748354704811, -11.569495816458623, -11.478524038252896, -11.386292814036862, -11.289781913656018, -11.198114725130194, -11.104132566769389, -11.01215017705774, -10.919598619692493, -10.824790826233828, -10.731309636422674]
#lnSum = [10.559672510284836, 10.370566932358992, 10.198057919524635, 9.996437242583644, 9.813633598660763, 9.628738358872065, 9.459848777439658, 9.274794204403829, 9.079058857055749, 8.901296222784037, 8.716351614895935, 8.537557930831401, 8.355664273649865, 8.1729231108762, 7.991068809131826, 7.80455855867819, 7.628637875355303, 7.445147468418159, 7.267862981237972, 7.087430250783669, 6.910472976896034, 6.733845113633998, 6.555580848723631, 6.380902442479547, 6.2076626362231835, 6.03196556243967, 5.86083760306589, 5.690552379153799, 5.520691117698525, 5.346051572939414, 5.183746623799561, 5.022444994728604, 4.851078235687854, 4.696994446503229, 4.536951860890872, 4.3836265286417975, 4.22887063954336, 4.075104635649463, 3.9306584048229456, 3.7831653682445356, 3.640837387595416, 3.4996801567118183, 3.3656491249747638, 3.235103380747408, 3.105918134810847, 2.979930391595545, 2.860926787749129, 2.744515398360849, 2.6321226914750193, 2.5247112396388864, 2.421407798617229, 2.321797805667118, 2.2279893837988327, 2.1372653429003092, 2.049922038222895, 1.972722409785107, 1.8927705354222228, 1.8185822002121288, 1.7501096300173573, 1.6871552594332246, 1.6260964617941376, 1.5708605273260763, 1.5184307176798686, 1.467260357324505, 1.4219994677818446, 1.3788131709108584, 1.3395417537717116, 1.3028592422763614, 1.2680624569587469, 1.2363446994385439]

def calcula_c(lnR, lnSum, MinValues, AMOSTRA_NAME):
    "Calcula a região linear da curva"
    lastError = 0
    for size in range(MIN_VALUES, len(lnR)):
        dots = [np.polyfit(lnR[0:size], lnSum[0:size], 1) for n in lnR]
        erro = 0
        experimentoIndex = 0
        simulacaoIndex = 0
        y = []
        for n in lnR:
            y.append(dots[0][0]*n + dots[0][1])
            
        for n in range(0, size):
            erro += (y[n] - lnSum[n])**2
            
        if (erro > lastError + 0.05 and lastError != 0):
            break
        lastError = erro
            
        #print(size, end="\t\t\t")
        #print(erro/size, end="\t\t\t")
        #print(dots[0][0])
    
    lastLinear = size
    c = dots[0][0]
    
    "Qual o trecho deve ser aproximado pela reta"
    dots = [np.polyfit(lnR[0:lastLinear], lnSum[0:lastLinear], 1) for n in lnR]
    
    "Início e fim da reta de aproximação"
    lnRDots = [np.linspace(lnR[0], lnR[-1])]
    
    z = dots[0]
    lnRDotsPlot = lnRDots[0]
    p = np.poly1d(z)
    
    "Plotagem"
    title_img = "C = " + str(round(c,5))
    labelLinear = "Y = " + str(round(c,5)) + "x + " + str(round(dots[0][1],5)) 
    fig, ax = plt.subplots()
    ax.plot(lnR, lnSum, '.')
    ax.plot(lnRDotsPlot, p(lnRDotsPlot), '-', label=labelLinear)
    ax.set(xlabel="Raio", ylabel="Np List", title=title_img)
    ax.grid()
    ax.legend()
    plt.gcf().set_size_inches(8, 5)
    fig.savefig("Export/" + AMOSTRA_NAME + "/0 - Calculo C.png", dpi = 300)
    plt.close(fig)

    return c
    