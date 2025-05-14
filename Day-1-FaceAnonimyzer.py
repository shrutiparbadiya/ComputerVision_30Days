import cv2
import numpy as np
import mediapipe as mp

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = face_detection.process(frameRGB)
        if out.detections is not None:
            for detection in out.detections:
                location_data = detection.location_data
                bbox = location_data.relative_bounding_box
                H, W, _ = frame.shape
                x,y,w,h, = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                x = int(x*W)
                y = int(y*H)
                w = int(w*W)
                h = int(h*H)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frameCrop = frame[y:y+h, x:x+w]
                blur = cv2.GaussianBlur(frameCrop, (101,101), 0)
                frame[y:y+h, x:x+w] = blur
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break