import cv2
import numpy as np
from numpy import save
from numpy import load

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


cap = cv2.VideoCapture(0)
count = 0
data=[]
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        data.append(face)
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1)==13 or count==100:
        break

cap.release()
cv2.destroyAllWindows()
print('Samples Colletion Completed ')
name=str(input('name : '))
d=[]
n=[]
#n=load('names.npy')
#n=n.tolist()
n.append(name)
#d=load('data.npy')
#d=d.tolist()
d.append(data)
save('data.npy',d)
save('names.npy',n)
