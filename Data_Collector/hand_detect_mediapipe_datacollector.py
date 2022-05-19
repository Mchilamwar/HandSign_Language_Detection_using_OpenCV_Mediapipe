# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 19:38:23 2022

@author: mchil
"""

import mediapipe as mp
import cv2
import pandas as pd


detect_model=mp.solutions.hands

drawing_mp=mp.solutions.drawing_utils
drawing_styles_mp=mp.solutions.drawing_styles


capture=cv2.VideoCapture(1)

empty=pd.read_csv('Dataset.csv',encoding='latin1')


with detect_model.Hands(model_complexity=1,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2,max_num_hands=1) as hands:
    
    
    
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
                    
                cord.append(6)
                   
                empty=empty.append(pd.DataFrame([cord],columns=empty.columns))
                image=cv2.putText(cv2.flip(image, 1),"Collecting Data",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(250,25,2.5),thickness=2)     
                cv2.imshow("hand",image)
        else:
                cv2.imshow("hand",cv2.flip(image,1))

        
        if cv2.waitKey(1) & 0xff==ord('e'):
            # cord['label']=1
            # print(chr(cv2.waitKey()))
            capture.release()
            break 

cv2.destroyAllWindows()            

empty[empty['label']==6]

# empty.to_csv("Dataset.csv",index=False,header=True)
