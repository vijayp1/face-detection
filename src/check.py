import cv2
import numpy as np
from numpy import load
from os import listdir
from os.path import isfile, join
from PIL import Image as img
def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi


def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


Labels =[]
for i in range(0,100):
    Labels.append(i)
Training_Data=load('data.npy')
names=load('names.npy')
l=len(names)


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data[0]), np.asarray(Labels))
cap = cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    face=face_extractor(frame)
    if face is not None:
        break
face = cv2.resize(face,(200,200))
face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
c=0
for i in range(0,l):
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data[i]), np.asarray(Labels))
    name=names[i]
    result = model.predict(face)
    if result[1] < 500:
        confidence = int(100*(1-(result[1])/300))
    if confidence > 82:
        c=1
        break
    del model
if c==1:
    print(name,"face is already recorded!")
else:
    print('new face please record using dataset')


cap.release()
cv2.destroyAllWindows()

