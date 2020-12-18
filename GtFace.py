import cv2
import numpy as np
import os 
cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('test\\haarcascade_frontalface_default.xml')

name = input('enter subject name : ')
frames = []
output = []
count = 0
trainCount = 10
while True:
    ret , frame = cap.read()

    if ret:
        faces = classifier.detectMultiScale(frame)
        for face in faces:
            x,y,w,h = face
            cut = cv2.cvtColor(cv2.resize(frame[y:y+h,x:x+w],(100,100)),cv2.COLOR_BGR2GRAY)
            cv2.imshow('face',cut)

        cv2.imshow("Main",frame)

        if count<trainCount:
            frames.append(cut.flatten())
            output.append([name])
            count +=1
            print(f'{count} pushed')
    if cv2.waitKey(1) == ord('q'):
        break;
    if cv2.waitKey(1) == ord('c'):
        count = 0
        
X = np.array(frames) 
y = np.array(output)

imdata = np.hstack([y,X])
fname = 'facedata.npy'
if(os.path.exists(fname)):
    old_data = np.load(fname)
    imdata = np.vstack([old_data,imdata])

np.save(fname,imdata)
         

cap.release()
cv2.destroyAllWindows()                            
