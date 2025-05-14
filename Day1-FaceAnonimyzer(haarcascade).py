import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

kernal = 101
while True:
    ret, frame = cam.read()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(frameGray, 1.1, 9)
    for (x, y, w, h) in faces:
        crop = frame[y:y+h, x:x+w]
        blur = cv2.GaussianBlur(crop, (kernal,kernal), 0)
        frame[y:y+h, x:x+w] = blur

    cv2.imshow('Webcam ', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break