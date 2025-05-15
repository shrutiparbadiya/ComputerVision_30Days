# import os
# import numpy as np
from ultralytics import YOLO
model = YOLO('yolov8n-cls.pt')

model.train(data='C:\\Users\\dell\\PycharmProjects\\ComputerVision_30Days\\Resources\\Weather-Analysis-main'
            , epochs = 20, imgsz = 64)