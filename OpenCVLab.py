#Author: Austin Johnson
#PRESS "Q" ON ANY OF THE FRAMES TO EXIT PROGRAM

import cv2
import numpy as np

kernel = np.ones((5,5), np.uint8)
hsv = 0
r = [0, 0, 0]
minimumHSV = np.array([0, 0, 0])
maximumHSV = np.array([0, 0, 0])

def passFun(x):
    pass

def onMouseClick(event, x, y, flags, pic):
    if event==cv2.EVENT_LBUTTONDOWN:
        global r
        r = hsv[y][x]
        print("At " + str(x) + ":X " + str(y)+ ":Y --" + str(r) + " Are the HSV values.")

cv2.namedWindow("Live Video", cv2.WINDOW_KEEPRATIO)

cv2.namedWindow("HSV Video", cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar("Hue", "HSV Video", 0, 255,passFun)
cv2.createTrackbar("Sat", "HSV Video", 0, 255,passFun)
cv2.createTrackbar("Val", "HSV Video", 0, 255,passFun)
cv2.createTrackbar("1", "HSV Video", 0, 255,passFun)
cv2.createTrackbar("2", "HSV Video", 0, 255,passFun)
cv2.createTrackbar("3", "HSV Video", 0, 255,passFun)
cv2.setMouseCallback("HSV Video", onMouseClick, hsv)

cv2.namedWindow("Black/White", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Erosion", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Dilation", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Diff", cv2.WINDOW_KEEPRATIO)
cv2.moveWindow("Live Video", 0, 0)
cv2.moveWindow("HSV Video", 0, 400)
cv2.moveWindow("Black/White", 420, 0)
cv2.moveWindow("Erosion", 420, 400)
cv2.moveWindow("Dilation", 840, 0)
cv2.moveWindow("Diff", 840, 400)

def updateScalars(hue, saturation, value):
    global r, minimumHSV, maximumHSV
    minimumHSV = np.array([r[0] - hue, r[1] - saturation, r[2] - value])
    maximumHSV = np.array([r[0] + hue, r[1] + saturation, r[2] + value])


cap = cv2.VideoCapture(0)
ret_val, frame = cap.read()

blankFrameA = np.float32(frame)
blankFrameB = np.float32(frame)
image = np.float32(frame)
difference = np.float32(frame)

while True:
    #Exit program if "Q" is entered
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    val, image = cap.read()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convert to HSV
    black_white = cv2.inRange(image, minimumHSV, maximumHSV) #create black and white image
    erode = cv2.erode(black_white, kernel, iterations=2)
    dilate = cv2.dilate(erode, kernel, iterations=2)
    blur = cv2.GaussianBlur(image,(5,5),0)
    cv2.accumulateWeighted(blur, blankFrameA, .4)
    res1 = cv2.convertScaleAbs(blankFrameA)
    difference = cv2.absdiff(image, res1)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    _, gray = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try: hierarchy = hierarchy[0]
    except: hierarchy = []

# Tries to draw rectangles. Severely decreases performance of program, uncomment to try.

#    for contour, hier in zip(contours, hierarchy):
#        (x,y,w,h) = cv2.boundingRect(contour)
#        if w > 20 and h > 20:
#            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
#
#        cv2.imshow('Diff',frame)

    cv2.drawContours(image, contours, -10, (0, 255, 0), 5)
    cv2.imshow("Diff", gray)
    cv2.imshow("Live Video", image)
    cv2.imshow("HSV Video", hsv)
    cv2.imshow("Black/White", black_white)
    cv2.imshow('Erosion', erode)
    cv2.imshow('Dilation', dilate)
    hue = cv2.getTrackbarPos('Hue', 'HSV Video')
    saturation = cv2.getTrackbarPos('Sat', 'HSV Video')
    value = cv2.getTrackbarPos('Val', 'HSV Video')
    updateScalars(hue, saturation, value)

cv2.destroyAllWindows()
