import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'images'
images = []
classNames = []

myList = os.listdir(path)

for cl in myList:
  currentImg = cv2.imread(f'{path}/{cl}')
  images.append(currentImg)
  classNames.append(os.path.splitext(cl)[0])
  
def findEncodings(images):
  encoded_list = []
  for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgEnconding = face_recognition.face_encodings(img)[0]
    encoded_list.append(imgEnconding)
  return encoded_list

def markAttendence(name):
  with open('attendence.csv', 'r') as f:
    myDataList = f.readline()
    print(myDataList)

encodeListKnow = findEncodings(images)

cap = cv2.VideoCapture(0)
while True:
  success, frame = cap.read()
  imgS = cv2.resize(frame, (0,0), None, 0.25, 0.25)
  imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
  
  facesCurrentFrame = face_recognition.face_locations(imgS)
  encodesCurrentFrame = face_recognition.face_encodings(imgS, facesCurrentFrame)
  
  for encodeFace,faceLoc in zip(encodesCurrentFrame,facesCurrentFrame):
    matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
    face_dist = face_recognition.face_distance(encodeListKnow, encodeFace)
    
    match_index = np.argmin(face_dist)
    
  
    if(matches[match_index]):
      name = classNames[match_index].upper()
      y1,x2,y2,x1 = faceLoc
      y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
      cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
      cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0), cv2.FILLED)
      cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255), 2)
      
      
      
  cv2.imshow('webcam', frame)
  
  cv2.waitKey(1)
  

