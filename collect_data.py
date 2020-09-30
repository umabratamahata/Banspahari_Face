# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 18:47:43 2020

@author: LENOVO
"""

import cv2
import urllib
import numpy as np

face_data = r"haarcascade_frontalface_default.xml"

classifier = cv2.CascadeClassifier(face_data)

url = "http://192.168.43.1:8080/shot.jpg"

data = []

while len(data)<100:
    image_from_url = urllib.request.urlopen(url)
    frame = np.array(bytearray(image_from_url.read()),np.uint8)
    frame = cv2.imdecode(frame,-1)
    
    faces = classifier.detectMultiScale(frame,1.5,5)
    
    if len(faces)>0:
        for x,y,w,h in faces:
            face_frame = frame[y:y+h,x:x+w].copy()
            cv2.imshow("only_face",face_frame)
            
            if len(data)<=100:
                print(len(data)+1,"/100")
                data.append(face_frame)
            else:
                break
            
    cv2.imshow("capture",frame)
    if cv2.waitKey(30)== ord("a"):
        break
    
cv2.destroyAllWindows()
    
if len(data)==100:
    name = input("enter the name : ")
    for i in range(100):
        cv2.imwrite("images/"+name+"_"+str(i+1)+".jpg",data[i])
        
    print("complete")

else:
    print("need more data")

