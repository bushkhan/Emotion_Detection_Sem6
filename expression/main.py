from tkinter import messagebox
from keras.models import load_model
from time import sleep
from keras_preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import testing
import random
import os
import tkinter as tk



face_classifier = cv2.CascadeClassifier(r'E:\Emotion_Detection_CNN\haarcascade_frontalface_default.xml')
classifier =load_model(r'E:\Emotion_Detection_CNN\model.h5'
)
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
            print("test 1")
            n = random.randint(1,1000)
            path1 = "E:/Emotion_Detection_Sem6/expression/testingaudios/recording"+str(n)+".wav"
            freq = 44100
            duration = 10
            print("test 2")
            recording = sd.rec(int(duration * freq),
            				samplerate=freq, channels=2)
            sd.wait()
            wv.write(path1, recording, freq, sampwidth=2)

            cap.release()
            cv2.destroyAllWindows()
            
            output = testing.run(path1)
            print("test 3")
            os.remove(path1)


            root = tk.Tk()
            root.overrideredirect(1)
            root.withdraw()
            messagebox.showinfo("Output","Predicted Emotion is: "+output+"")
            print("test 4")

            break

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break



cap.release()
cv2.destroyAllWindows()