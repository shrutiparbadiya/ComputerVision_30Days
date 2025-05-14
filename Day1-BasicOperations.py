from pickletools import uint8

import cv2
import numpy as np

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

img = cv2.imread('Resources/bird.jpg')

################# resizing #################

imgResize = cv2.resize(img, (552,364))

################# Crop #################

imgCrop = img[50:100,100:200]

#Color_conversion
imgRGB = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)
imgGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
imghsv = cv2.cvtColor(imgResize, cv2.COLOR_BGR2HSV)

################# Blur Image #################

k_size = 23
imgBlur = cv2.blur(img, (k_size,k_size))
imgGaussianblur = cv2.GaussianBlur(img, (k_size,k_size), 2)
imgMedian = cv2.medianBlur(img, k_size)

################# clean noisy Image #################

kernal = 7
image = cv2.imread('Resources/dog.jpg')
imageBlur = cv2.blur(image, (kernal, kernal))
imageGaussianblur = cv2.GaussianBlur(image, (kernal, kernal), 2)
imageMedian = cv2.medianBlur(image, kernal)
Stacked = stackImages(0.4, [[image,imageBlur],
                      [imageGaussianblur,imageMedian]])

# ################# Image Thresholding #################

bear = cv2.imread('Resources/bear.jpg')
bearGray = cv2.cvtColor(bear, cv2.COLOR_BGR2GRAY)
bearThreshold = cv2.threshold(bearGray, 80, 255, cv2.THRESH_BINARY)[1]
bearBlur = cv2.medianBlur(bearThreshold, 5)

writing = cv2.imread('Resources/writing.jpg')
writingGray = cv2.cvtColor(writing, cv2.COLOR_BGR2GRAY)
writingThreshold = cv2.adaptiveThreshold(writingGray, 140, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,30)
cv2.imshow('writing', writingThreshold)


################# Edge Detection #################

player = cv2.imread('Resources/player.png')
playerCanny = cv2.Canny(player, 100, 200)
playerDialate = cv2.dilate(playerCanny, np.ones((5,5), dtype = 'int32'),2)
plyareErode = cv2.erode(playerDialate, np.ones((5,5), dtype = 'int32'),0)


################# Draw on Image #################

whiteboard = cv2.imread('Resources/whiteboard.jpg')
whiteboardResize = cv2.resize(whiteboard, (552,364))
draw = whiteboardResize.copy()

cv2.rectangle(draw, (150, 100), (250, 200), (255, 0, 0), 2)
cv2.circle(draw, (300, 175), 20, (0, 0, 255), cv2.FILLED)
cv2.putText(draw, "Hello World", (300,100),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
cv2.line(draw, (150, 100), (250, 200), (0, 0, 0), 2)

################# Contour #################

birds = cv2.imread('Resources/birds.png')
birdsGray = cv2.cvtColor(birds, cv2.COLOR_BGR2GRAY)
birdsThreshold = cv2.threshold(birdsGray, 127, 255, cv2.THRESH_BINARY_INV)[1]
contours, hierarchy = cv2.findContours(birdsThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) > 200:
        cv2.drawContours(birds, cnt,-1, (0,0,255),2)

        x1,y1,w,h = cv2.boundingRect(cnt)

        cv2.rectangle(birds,(x1,y1),(x1+w,y1+h),(0,255,0),2)



# ################# Visualize Image #################

imgStacked = stackImages(1.5,[[img,imgCrop,imgBlur],
                              [imgGray,imgRGB,imghsv]])

cv2.imshow('Color Comparison', imgStacked)
cv2.imshow('Noise Cleaned', Stacked)
cv2.imshow('Threshold', bearBlur)
cv2.imshow('birds', birds)
cv2.imshow('playerEdge', playerCanny)
cv2.imshow("Draw", draw)

###################################################

# cv2.imshow("median", imgMedian)
# cv2.imshow('imgBlur', imgGaussianblur)
# cv2.imshow('image', img)
# cv2.imshow('imageResize', imgResize)
# cv2.imshow('imageCrop', imgCrop)
# cv2.imshow('imageGray', imgGray)
# cv2.imshow('imagehsv', imghsv)
# cv2.imshow('imageRGB', imgRGB)
# cv2.imshow('Dialate', playerDialate)
# cv2.imshow('Erode', plyareErode)


cv2.waitKey(0)
