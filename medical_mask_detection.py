# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 

@author: DEMİR
"""

import cv2 

faceCascade = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')
mouthCascade = cv2.CascadeClassifier("cascades\haarcascade_mcs_mouth.xml")


camera = cv2.VideoCapture(0 , cv2.CAP_DSHOW)

while True:
    
    _, square = camera.read()
    square = cv2.flip(square , 1)
    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
    
    faces  = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(20, 20))
    smiles = mouthCascade.detectMultiScale(gray,scaleFactor=1.95,minNeighbors=5,minSize=(20, 20)) 
    
    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(0,85,255),2)
    
    for (x,y,w,h) in smiles:
       cv2.rectangle(gray,(x,y),(x+w,y+h),(0,255,255),2)
       
    if(len(faces) != 0 and len(smiles) != 0):
        cv2.putText(gray, "WEAR A MASK", (30,30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        
    if(len(faces) != 0 and len(smiles) == 0):
        cv2.putText(gray, "MASK DETECTED", (30,30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,0), 1, cv2.LINE_AA)

       
     
    cv2.imshow("Do you wearing a mask ?",gray)
    
    k = cv2.waitKey(1) & 0xff
    if k == 27 or k==ord('q'):
        break




camera.release()
cv2.destroyAllWindows()
