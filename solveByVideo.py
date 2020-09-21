import cv2
from Extractor import *
from solveIt import *
from outImageGenerator import outputImageGenerator
import copy
from compareSudoku import comparer

import cv2

cap = cv2.VideoCapture("sudokus.avi")
while not cap.isOpened():
    cap = cv2.VideoCapture("sudokus.avi")
    cv2.waitKey(1000)
    print "Okuyamadim"

pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
while True:
    flag, frame = cap.read()
    if flag:
        # The frame is ready and already captured
        cv2.imshow('video', frame)
        pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        print str(pos_frame)+" frames"
    else:
        # The next frame is not ready, so we try to read it again
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
        print "frame is not ready"
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:
        break
    if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        break