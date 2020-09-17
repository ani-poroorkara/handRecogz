#Import Libs 
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import numpy as np
from sklearn.metrics import pairwise
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras import models
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import glob
import operator
import pyautogui as pg


#Declare Background as none
bg = None


#Load Model
model = load_model('gesture.h5')


#Define the background  
def running_avg(image, accumWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, accumWeight)


def segment(image):
    global bg

    #Pre processing the image before prediction
    diff = cv2.absdiff(bg.astype("uint8"), image)
    gray = cv2.GaussianBlur(diff, (5, 5), 0)
    thresholded = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    
    #Find Contours
    cnts, heriar = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #Display processing
    cv2.imshow("Difference",diff)
    cv2.imshow("Gray",image)
    cv2.imshow("Thesholded", thresholded)
    
    #Process contours
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def recog(thresholded):    
    
    #Preprocessing and reshaping data 
    thresholded = cv2.resize(thresholded,(56,56))
    x_test_data = np.array(thresholded, dtype = 'float32')
    x_test_data = x_test_data.reshape((1, 56, 56, 1))
    x_test_data = x_test_data / 255

    #Predict the gesture
    prediction = model.predict(x_test_data)
    maxv = np.amax(prediction, axis=1)
    pos = np.where(prediction == maxv)

    #List of entire gestures
    gesture_list = {'Blank': prediction[0][0], 
                  'Palm': prediction[0][1], 
                  'L': prediction[0][2],
                  'Blank': prediction[0][3],
                  'Fist Moved': prediction[0][4],
                  'Down': prediction[0][5],
                  'Index': prediction[0][6],
                  'Ok': prediction[0][7],
                  'Palm Moved': prediction[0][8],
                  'C': prediction[0][9],
                  'Heavy': prediction[0][10],
                  'Hang': prediction[0][11],
                  'Two': prediction[0][12],
                  'Three': prediction[0][13],
                  'Four': prediction[0][14],
                  'Five': prediction[0][15],
                  'PalmReverse': prediction[0][16],
                  'Up': prediction[0][17],
    }

    #Sort the list as to find element with highest probabilistic value
    gesture_list = sorted(gesture_list.items(), key=operator.itemgetter(1), reverse=True)
    
    #Print for debugging 
    print(gesture_list[0][0])

    #Return value to main program
    return gesture_list[0][0]


if __name__ == "__main__":
    camera = cv2.VideoCapture(0)

    #Set initial number of frames as 0
    numframe = 0

    #Define coordinates of Region of Interest
    t, r, b, l = 60, 350, 285, 590

    #Set caliberation as not done
    calibrated = False

    print("*********************************************************************")
    print("<<<<<<<<<< HAND RECOGZ DEVICE >>>>>>>>>>")
    print("*********************************************************************")
    print("GitHub Link : https://github.com/ani-poroorkara/handRecogz")
    print("Stay clear of the ROI while background gets caliberated")

    while(True):
        
        #Grab the fram from camera
        (grabbed, frame) = camera.read()

        #Resize width 
        frame = imutils.resize(frame, width=700)
        clone = frame.copy()

        #Get dimensions of frame
        (height, width) = frame.shape[:2]

        #Fix the Region of Interest
        roi = frame[t:b, r:l]

        #convert the ROI to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        #Caliberate background for the first 100 frames
        if numframe < 100:
            running_avg(gray, 0.5)
            if numframe == 1:
                print("Please wait while the background caliberates")
            elif numframe == 99:
                print ("Caliberation Successfull")
        
        else:
            #Receive the processed image 
            hand = segment(gray)
            
            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (r, t)], -1, (139, 0, 139))

                #Find the gesture
                prediction = recog(thresholded)

                #Experimental pyAutoGUI to do system function
                if prediction =="L":
                    pg.screenshot('ss.png')

                #Show the prediction on the video frame 
                cv2.putText(clone, prediction, (200, 45), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,255), 2)

        #Show the ROI on frame 
        cv2.rectangle(clone, (l, t), (r, b), (0,0,0), 2)

        #Increase number of frames
        numframe += 1

        #Show the video feed
        cv2.imshow("Hand Rekogz", clone)

        #Exit the code
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
