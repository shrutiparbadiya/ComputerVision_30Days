import cv2
import numpy as np

mycolor = [110,80,100,130,255,255] #purple

def findColor(frame,mycolor):
    imghsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(mycolor[0:3])
    upper = np.array(mycolor[3:6])
    print(lower)
    mask = cv2.inRange(imghsv, lower, upper)
    getContour(mask,frame)
    cv2.imshow("HSV", imghsv)
    cv2.imshow("Mask", mask)


cam = cv2.VideoCapture(0)

def getContour(mask,frame):
    contour, hirerchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area > 500:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)

while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        print("❌ Failed to grab frame. Exiting...")
        break
    findColor(frame,mycolor)
    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()