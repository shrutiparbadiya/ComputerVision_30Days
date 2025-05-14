import os
from tkinter import Image

import cv2
#######################################################
#Image Operations
#######################################################
image_path = "Resources/bird.jpg"

##### Read Image #####

image = cv2.imread(image_path)

##### Write Image #####

cv2.imwrite('Resources/saved/bird.jpg', image)

##### Visualize Image #####

cv2.imshow('image', image)
cv2.waitKey(0)

#######################################################
#Video Operations
#######################################################

video_path = "Resources/video.mp4"

##### Read Video #####

cap = cv2.VideoCapture(video_path)

##### Visualize Video #####

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#######################################################
#Webcam Operations
#######################################################

##### Read Webcam #####

webcam_path = cv2.VideoCapture(1)

##### Visualize Webcam #####
while True:
    ret, img = webcam_path.read()
    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_path.release()
cv2.destroyAllWindows()

