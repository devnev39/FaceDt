import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('test\\haarcascade_frontalface_default.xml')
model = KNeighborsClassifier()

data = np.load('test\\facedata.npy')
X = data[:,1:].astype(int)
y = data[:,0]
model.fit(X,y)

while True:
    ret , frame = cap.read()

    if ret:
        faces = classifier.detectMultiScale(frame)
        for face in faces:
            x,y,w,h = face
            cut = cv2.cvtColor(cv2.resize(frame[y:y+h,x:x+w],(100,100)),cv2.COLOR_BGR2GRAY)
            out = model.predict([cut.flatten()])
            cv2.rectangle(frame,(x,y),(x+w,y+h),[0,255,0],thickness=2)
            cv2.putText(frame,str(out[0]),(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,[0,0,255],thickness=1)

        cv2.imshow('Main',frame)

    if cv2.waitKey(1) == ord('q'):
        break           

cap.release()
cv2.destroyAllWindows()        
     
            
