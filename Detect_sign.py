# -*- coding: utf-8 -*-
"""
Created on Sat May  7 10:43:34 2022

@author: mchil
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 19:38:23 2022

@author: mchil
"""

import mediapipe as mp
import cv2
import pandas as pd
import pickle as pkl
import tensorflow as tf
import keras as ks
import numpy as np


detect_model=mp.solutions.hands

drawing_mp=mp.solutions.drawing_utils
drawing_styles_mp=mp.solutions.drawing_styles



####################################################

trained_model=ks.models.load_model("D:\\IVY Batches\\MY ML DL Projects\\CNN\\Hand Signs\\trained_model")

with open("D:\\IVY Batches\\MY ML DL Projects\\CNN\\Hand Signs\\trained_scaler_model.pkl",'rb') as fil:
     trained_scaler_model=pkl.load(fil)
     
# Detection Function
def detect_sign(inp):
    inp=np.array(inp).reshape(1,-1)
    inp=trained_scaler_model.transform(inp)
    prediction=trained_model.predict(inp)
    
    label=prediction.argmax()
    
    lis=['A','B','C','D','E','F','G','H',
         'I','J']
    
    return lis[label]
        
    





####################################################

capture=cv2.VideoCapture(0)




with detect_model.Hands(model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,max_num_hands=1) as hands:
    
    
    
    while capture.isOpened():
        _,image=capture.read()
        
        image.flags.writeable=False
        
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        detection=hands.process(image)
        
        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        if detection.multi_hand_landmarks:
            
            for landmarks in detection.multi_hand_landmarks:
                drawing_mp.draw_landmarks(
                                image,
                                landmarks,
                                detect_model.HAND_CONNECTIONS,
                                drawing_styles_mp.get_default_hand_landmarks_style(),
                                drawing_styles_mp.get_default_hand_connections_style())
                # Co-ordinates list
                cord=[]
                
                for i in range(21):
                    cord.append(landmarks.landmark[i].x)
                    cord.append(landmarks.landmark[i].y)
                    cord.append(landmarks.landmark[i].z)
                    
                
                
                # Prediction 
                pred_sign=detect_sign(cord)
                
                image=cv2.putText(cv2.flip(image, 1),"This Sign is "+pred_sign,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(250,25,2.5),thickness=2,lineType=cv2.LINE_AA)     
                image=cv2.flip(image, 1)
        cv2.imshow("hand",cv2.flip(image, 1))
        
        if cv2.waitKey(1) & 0xff==ord('e'):
            # cord['label']=1
            # print(chr(cv2.waitKey()))
            capture.release()
            break 

cv2.destroyAllWindows()            




