import cv2
import numpy as np
import face_recognition

img = face_recognition.load_image_file('images/elon-musk.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img1 = face_recognition.load_image_file('images/elon_musk1.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

face_location = face_recognition.face_locations(img)[0]
face_location1 = face_recognition.face_locations(img1)[0]

encode = face_recognition.face_encodings(img)[0]
encode1 = face_recognition.face_encodings(img1)[0]

cv2.rectangle(img,(face_location[3], face_location[0]),(face_location[1], face_location[2]), (255,0,255), 2)
cv2.rectangle(img1, (face_location1[3], face_location1[0]),(face_location1[1], face_location1[2]), (255,0,255), 2)

result = face_recognition.compare_faces([encode], encode1)
face_distance = face_recognition.face_distance([encode], encode1)
cv2.putText(img1,f'{result} {round(face_distance[0], 2)}',(50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)


cv2.imshow('Image', img)
cv2.imshow('Image1', img1)

cv2.waitKey(0)