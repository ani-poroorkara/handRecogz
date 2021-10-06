import cv2
import imutils
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


bg = None
model = load_model('gesture.h5')


def recog(thresholded):    

    thresholded = cv2.resize(thresholded,(320,120))
    x_test_data = np.array(thresholded, dtype = 'float32')
    x_test_data = x_test_data.reshape((1, 120, 320, 1))
    x_test_data = x_test_data / 255
    result = model.predict(x_test_data)
    return result

def run_avg_bg(image):
    global bg
    if bg is None:
        bg = image
        return

if __name__ == "__main__":

    camera = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    top, right, bottom, left = 60, 350, 285, 590
    num_frames = 0
    calibrated = False

    while(True):
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 50:
            run_avg_bg(gray)
            if num_frames == 1:
            	print (">>>Please wait! Program is calibrating the background...")
            elif num_frames == 49:
                print (">>>Calibration successfull.....")

        else:
            
            diff = cv2.absdiff(bg, gray)
            cv2.imshow("diff = grey - bg",diff)
            thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresholded, None, iterations=2)


            cv2.imshow("Thesholded", thresholded)
            cv2.imshow("Thesholded_dilated", thresh)
            prediction = recog(thresholded)

            maxv = np.amax(prediction, axis=1)
            pos = np.where(prediction == maxv)

            gesture_list = {'palm': prediction[0][0], 
                  'L': prediction[0][1], 
                  'fist': prediction[0][2],
                  'fist_moved': prediction[0][3],
                  'thumb': prediction[0][4],
                  'index': prediction[0][5],
                  'ok': prediction[0][6],
                  'palm_moved': prediction[0][7],
                  'c': prediction[0][8],
                  'down': prediction[0][9]
                  }
            #Top prediction
            gesture_list = sorted(gesture_list.items(), key=operator.itemgetter(1), reverse=True)
            
            print(gesture_list[0][0])
            cv2.putText(clone, gesture_list[0][0], (200, 45), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,255), 2)
            cv2.imshow("Thesholded", thresholded)
        
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        cv2.imshow("Video Feed", clone)
        
        num_frames += 1

        # observe the keypress by the user; if the user has pressed "q", then stop looping
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break

    # free up memory
    camera.release()
    cv2.destroyAllWindows()