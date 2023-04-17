from ultralytics import YOLO
import cv2 as cv
import math
import matplotlib.pyplot as plt
import time
from sort import *

# dependencies: filterpy / scikit-image / lap

model = YOLO("YoloWeights/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# read image
image = cv.imread('Masks/carimg.jpg')
# plt.imshow(img)
# plt.show()
blank = np.zeros(image.shape[:2], dtype='uint8')
polygons = np.array([ [(30, 250),(250, 250),(270,150), (100,150)] ])
mask = cv.fillPoly(blank.copy(), polygons, 255)
masked = cv.bitwise_and(image, image, mask=mask)
# cv.imshow("mask", masked)
# cv.waitKey(0)
cap = cv.VideoCapture('Videos/carsVideo.mp4')
new_frame_time = 0
prev_frame_time = 0
while True:
        new_frame_time = time.time()
        success, img = cap.read()
        imgRegion = cv.bitwise_and(img, img, mask=mask)
        results = model(imgRegion, stream=True)
        # initial a detection array
        detections = np.empty((0, 5))
        for r in results:
                boxes = r.boxes
                for box in boxes:
                        # extract bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        # draw bounding box
                        # cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
                        # get Confidence
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        # get Class Name
                        cls = int(box.cls[0])
                        currentClass = classNames[cls]
                        # we're only interested in car/truck/bus with probability>0.4
                        # i.e. if it is a motorbike, it will not show the tag
                        if (currentClass=="car" or currentClass=='truck' or currentClass=="bus") and conf>0.4:
                                currentArray = np.array([x1,y1,x2,y2, conf])
                                detections = np.vstack((detections, currentArray))
                                # show text
                                # cv.putText(img, f'{classNames[cls]} {conf}', (max(0, x1), max(20, y1)), cv.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255), 1)
                                # when detected such a vehicle, save to an array
                                currentArray = np.array([x1,y1,x2,y2, conf])
                                # append to the list, in np use stack
                                detections = np.vstack((detections, currentArray))
        # update the results of tracker        
        resultsTracker = tracker.update(detections)

        for result in resultsTracker:
                x1,y1,x2,y2,id = result
                print(result)
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                # draw the tracking bounding box  
                cv.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
                cv.putText(img, f'{int(id)}', (max(0, x1), max(20, y1-5)), cv.FONT_HERSHEY_TRIPLEX, 0.7, (209, 80, 255), 1)

        # calculate time of each frame
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        # print(fps)
        # show the frames
        cv.imshow('img', img)
        # break the loop by press 'q' 
        if cv.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv.destroyAllWindows()

# python yolo-project-track.py

