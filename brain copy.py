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

# Function - To find the running average over the background
def run_avg(image, accumWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, 0.1)

def segment(image, threshold=30):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)
    cv2.imshow("diff = grey - bg",diff)
    cv2.imshow("grey",image)
    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    cnts, heriar = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heriar = None
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def count(thresholded, segmented):
    
    thresholded = cv2.resize(thresholded,(56,56))
    x_test_data = np.array(thresholded, dtype = 'float32')
    x_test_data = x_test_data.reshape((1, 56,56, 1))
    x_test_data = x_test_data / 255
    result = model.predict(x_test_data)
    return result

# Main function
if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    # region of interest (ROI) coordinates
    top, right, bottom, left = 60, 350, 285, 590

    # initialize num of frames
    num_frames = 0

    # calibration indicator
    calibrated = False

    # keep looping, until interrupted
    while(True):
        # get the current frame 
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our weighted average model gets calibrated
        if num_frames < 100:
            run_avg(gray, 0.5)
            if num_frames == 1:
            	print (">>>Please wait! Program is calibrating the background...")
            elif num_frames == 99:
                print (">>>Calibration successfull. ...")
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:

                (thresholded, segmented) = hand
                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                # count the number of fingers
                prediction = count(thresholded, segmented)
                maxv = np.amax(prediction, axis=1)
                pos = np.where(prediction == maxv)

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
                #Top prediction
                gesture_list = sorted(gesture_list.items(), key=operator.itemgetter(1), reverse=True)
                print(gesture_list[0][0])

                cv2.putText(clone, gesture_list[0][0], (200, 45), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,255), 2)
                # show the thresholded image
                cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user has pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()