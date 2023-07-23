# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 22:32:40 2020

@author: user1
"""

import cv2 
import numpy as np 
import math  
import matplotlib.pyplot as plt 
import pandas as pd
import os
import json
import gc


# THE FILES MUST BE LIKE THIS:
# \LS2                  DIRECTORY
# \LS2\LS2.mp4          VIDEO FILE WITH THE SAME NAME AS THE DIRECTORY
# \LS2\LS2.json         PROPETIES FILE WITH THE SAME NAME AS THE DIRECTORY
# \LS2\Arduino.txt      OPTIONAL FILE WITH ARDUINO EXPORT TO THIS TEST

"""
=== SELECT WHAT THE PROGRAM WILL DO
"""

last_circle_left = 0
last_circle_right = 0


DS_list = ['DS15', 'DS16']
DS_list = ['DS15']

img_save = 0
for VIDEO in DS_list:
    print(VIDEO)
    
    """
    === VIDEO VARIABLES
    """
    videoName               = VIDEO
    videoExtention          = '.mp4'
    fps_export              = 60;
    count_frames            = 0;
    rMin                    = 0;            # CIRCLE RADIUS
    rMax                    = 0;
    minDistance             = 640;          # MINIMUM DISTANCE BETWEEN 2 CIRCLES
    ard_read                = 0;            # CORRELATION BETWEEN VIDEO_FRAME AND ARDUINO READ
    vid_frame               = 0;
    
    circ_param_1            = 0;
    circ_param_2            = 0;
    
    """
    === DISTANCE VARIABLES
    """
    squares                 = {}            # SQUARES POSITION - LU -> Left Up, RD -> Right Down
    
    
    """
    === SET THE FILE NAMES
    """
    
    DIRECTORY               = videoName + "\\Export\\"; # PC
    
    #VIDEO_FILE              = videoName + '\\' + videoName + '_cut' + videoExtention
    JSON_FILE               = videoName + '\\Parameters_' + videoName + '.json'
    CSV_FILE                = videoName + '\\Distances_' + videoName + '.csv' 
    ARDUINO_FILE            = videoName + '\\' + 'Arduino.txt'
    
    
    """
    === CREATE THE DIRECTORY TO SAVE THE IMAGES
    """
    if not os.path.isdir(videoName + "\Export"):
        os.mkdir(videoName + "\Export")
    if not os.path.isdir(videoName + "\Export\FRAMES"):
        os.mkdir(videoName + "\Export\FRAMES")
    if not os.path.isdir(videoName + "\Export\ANALYZED"):
        os.mkdir(videoName + "\Export\ANALYZED")
    if not os.path.isdir(videoName + "\Export\ANALYZED_FRAMES"):
        os.mkdir(videoName + "\Export\ANALYZED_FRAMES")
    
    """
    === IMPORT THE VARIABLES TO THIS VIDEO
    """
    if os.path.isfile(JSON_FILE):
        with open(JSON_FILE, "r") as json_file:
            my_dict = json.load(json_file)
        locals().update(my_dict)
    else:
        print("Não foi possível carregar as variáveis, o software será fechado")
        os._exit(1)
    del json_file, my_dict
    
    """
    === CLEAR THE CSV FILE
    """
    if False:
        open(CSV_FILE, "w");                # Create a file again
    
    
    """
    ===========
        ===========
            ===========
                ===========
                Image prepare.
                ===========
            ===========
        ===========
    ===========
    """
    gc.collect()
    dist_list = []
    
    n = 0
    while True:
        n += 1
        if not os.path.isfile(videoName + '\\Export\\FRAMES\\frame_' + str(n) + '.jpg'):
            break
        
    x, y, r = 0, 0, 0
    for i in range(0, n):
    #for i in range(70, n, 5):
        #if not i % 50:
        print(i)
            
        distance = 0
        
        img = cv2.imread(DIRECTORY + 'FRAMES/frame_'+str(i)+'.jpg', cv2.IMREAD_COLOR)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #ret, imgf = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        imgf = cv2.adaptiveThreshold(imgGray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                  cv2.THRESH_BINARY,25,2)
    
        # Crop the image
        crop_left = imgf[squares["LU"][1]:squares["LD"][1], squares["LU"][0]:squares["LD"][0]]
        crop_right = imgf[squares["RU"][1]:squares["RD"][1], squares["RU"][0]:squares["RD"][0]]
        
        # Create a new image
        height, width, channels = img.shape
        crop_img = np.zeros((height,width), np.uint8)
        crop_img[:,:] = (0)      # (B, G, R)
        
        # Put the cops in the new image
        crop_img[squares["LU"][1]:squares["LD"][1], squares["LU"][0]:squares["LD"][0]] = crop_left
        crop_img[squares["RU"][1]:squares["RD"][1], squares["RU"][0]:squares["RD"][0]] = crop_right
        
        if (last_circle_left + last_circle_right) != 0:
            rMin = int(min(last_circle_left,last_circle_right) * .95)
            rMax = int(max(last_circle_left,last_circle_right) * 1.05)

        detected_circles = cv2.HoughCircles(crop_img, cv2.HOUGH_GRADIENT, 1, 4*rMax, 
                                            param1 = circ_param_1, param2 = circ_param_2,
                                            minRadius = rMin, maxRadius = rMax)
        
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))
            for circle in detected_circles[0, :]: 
                x, y, r = circle[0], circle[1], circle[2]
                cv2.circle(img, (x, y), r, (0, 255, 0), 4) 
                plt.text(x, y+100, str(r), color='b', fontsize=24)
        
        cv2.rectangle(img, tuple(squares["LU"]), tuple(squares["LD"]), (255,0,0), 5)
        cv2.rectangle(img, tuple(squares["RU"]), tuple(squares["RD"]), (255,0,0), 5)
        
        if img_save:
            fig = plt.figure(figsize=(16,9))
            plt.imshow(img,cmap = 'gray')
            plt.savefig(DIRECTORY + 'ANALYZED/_'+str(i)+'_radius.jpg', dpi = 120)
            
            fig = plt.figure(figsize=(16,9))
            plt.subplot(1,1,1), plt.imshow(imgf,cmap = 'gray')
            plt.savefig(DIRECTORY + 'ANALYZED/+'+str(i)+'_gray.jpg', dpi = 120)
            
            fig = plt.figure(figsize=(16,9))
        
        # If there is a circle in the image
        if detected_circles is not None:
            # Convert the circle parameters a, b and r to integers. 
            detected_circles = np.uint32(np.around(detected_circles))
          
            x1 = 0; x2 = 0; y1 = 0; y2 = 0;
            for circle in detected_circles[0, :]: 
                x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
                cv2.circle(imgf, (x, y), r, (0, 255, 0), 4) 
                if img_save:
                    plt.text(x, y+100, str(r), color='b', fontsize=24)
                draw = False
                
                # If the center of the circle is inside of the square, save the coordenates
                if ((x > squares['LU'][0] and x < squares['LD'][0]) and (y > squares['LU'][1] and y < squares['LD'][1])):
                    last_circle_left = r
                    x1 = x
                    y1 = y
                    draw = True
                    
                # If the center of the circle is inside of the square, save the coordenates
                if ((x > squares['RU'][0] and x < squares['RD'][0]) and (y > squares['RU'][1] and y < squares['RD'][1])):
                    last_circle_right = r
                    x2 = x
                    y2 = y
                    draw = True
            
            # Calculate the distance
            if (x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0):
                imgf = cv2.line(imgf, (x1, y1), (x2, y2), (0, 255, 255), 5)
                distance = round(math.sqrt(abs(x2-x1)**2 + abs(y2-y1)**2), 2)                
                if img_save:
                    plt.text(x1+450, y1 - 150, str(distance), color='b', fontsize=24)
              
        
        if img_save:
            plt.imshow(imgf,cmap = 'gray')
            plt.savefig(DIRECTORY + 'ANALYZED_FRAMES/'+str(i)+'.jpg', dpi = 120)  
            
        #plt.close("all")
        dist_list.append(distance)

        #del img, detected_circles, fig
    
    #fig = plt.figure(figsize=(16,9))
    #plt.plot(linearSpace, dist_list)
    #plt.savefig(DIRECTORY + 'ANALYZED_FRAMES/__dist.jpg', dpi = 120)  
    
    del x, y, r, x1, x2, y1, y2, n, crop_img, crop_left, crop_right
    
    

    df = pd.read_csv(ARDUINO_FILE, sep="\t", index_col=0)
    
    df['DISTANCE'] = dist_list
    df = df.drop(columns = ['READ_HEAT'])
    
    # ESSA DESGRAÇA DO INFERNO NÃO EXPORTA CERTO, VERIFICAR MANUALMENTE
    
    df.to_csv(VIDEO + '\out.csv', index=False) 
    df = df.replace("", np.nan)
    
    fig = plt.figure(figsize=(16,9))
    plt.plot(df['READ_EXT_TEMP'], df['DISTANCE'])
    plt.savefig(VIDEO + '/DILAT.jpg', dpi = 120)
    fig = plt.figure(figsize=(16,9))
    plt.plot(df['READ_EXT_TEMP'])
    plt.savefig(VIDEO + '/Temperature.jpg', dpi = 120)
    fig = plt.figure(figsize=(16,9))
    plt.plot(df['DISTANCE'])
    plt.savefig(VIDEO + '/Distance.jpg', dpi = 120)
    fig = plt.figure(figsize=(16,9))
    plt.plot(df['READ_NUMBER'])
    plt.savefig(VIDEO + '/Time.jpg', dpi = 120)
    
    plt.close("all")