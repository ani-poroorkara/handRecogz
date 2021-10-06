from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

tracker = cv2.TrackerCSRT_create()


initBB = None
fps = None
camera = cv2.VideoCapture(0)

while(True):
    k, frame = camera.read()
    # if not k:
    #     print 'Cannot read video file'
    #     sys.exit()

    frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)

    if initBB is not None:
		# grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)
		# check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # update the FPS counter
        fps.update()
        fps.stop()


    # show the output frameqq
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()
    
    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

cv2.destroyAllWindows()

