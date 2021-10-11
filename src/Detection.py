import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from numpy import load
from PIL import Image as img
import os,shutil

Labels = []

for i in range(100):
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

data=load('data.npy')
names=load('names.npy')

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
for i in range(len(names)):
    if not os.path.exists(names[i]):
        os.mkdir(names[i])
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
def check(model,face):
    result=model.predict(face)
    if result[1] < 500:
        confidence = int(100*(1-(result[1])/300))
    if confidence > 82:
        return(1)
    else:
        return(0)
data_path = "G:/assignment/sem 4/projects/AI/facial_recognition-main/test/"
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
for i in range(0,len(onlyfiles)):
    file=data_path+onlyfiles[i]
    im=img.open(file)
    frame=np.array(im)
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        for j in range(0,len(names)):
            Training_Data=data[j]
            name=names[j]
            model = cv2.face.LBPHFaceRecognizer_create()
            model.train(np.asarray(Training_Data), np.asarray(Labels))
            result=check(model,face)
            if result==1:
                shutil.move(file,name)
                break
            if(j==len(names)-1):
                print(onlyfiles[i],"  -  unknown")
            del model,j,result,name

    except:
        print(onlyfiles[i],"  -  face not found")
    del file,im,frame,image,face

