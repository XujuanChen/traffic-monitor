from ultralytics import YOLO
import cv2 as cv
import math
import matplotlib.pyplot as plt
import time
from sort import *
from deep_sort_realtime.deepsort_tracker import DeepSort
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

def rescaleFrame(frame, scale=2):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# resize an image
# img = cv.imread('Masks/carimg.jpg')
# resized_image = rescaleFrame(img)
# cv.imshow('Cat Resized', resized_image)
# cv.waitKey(0)

# tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# read image
ori_image = cv.imread('Masks/carimg.jpg')
image = rescaleFrame(ori_image)
# plt.imshow(image)
# plt.show()  # heihts[0:720], width[0:1280]

# cv.imshow("mask", masked)
blank3d = np.zeros_like(image) # the same 3D shape of image filled with zeros
banner = cv.rectangle(blank3d.copy(), (900,0), (1282,720), (255,0,0),-1)
# cv.imshow("banner",banner)
# cv.waitKey(0)

points1 = [220,300,500,300]
points2 = [120,450,500,450]
blank = np.zeros(image.shape[:2], dtype='uint8')
polygon = np.array([ [ (10, 580),(530, 580), (510,180),(320,180) ] ])
mask = cv.fillPoly(blank.copy(), polygon, 255)

# read video
cap = cv.VideoCapture('Videos/carsVideo.mp4')
# new_frame_time = 0
# prev_frame_time = 0
countSet=set()
vehicles_enter_time = {}
vehicles_leave_time = {}
vehicles_elaps_time = {}
speed = []
# cv.imshow("combine", mask_combine)
while True:
        # new_frame_time = time.time()
        success, frame = cap.read()
        img = rescaleFrame(frame)

        imgRegion = cv.bitwise_and(img, img, mask=mask)
        # draw the area
        cv.polylines(img, polygon, True, (0,0,255), 3)

        results = model(imgRegion, stream=True)
        # put side banner mask 
        # img = cv.bitwise_and(img, img, mask=banner)
        frame = np.copy(img)
        img = cv.addWeighted(frame,0.6,banner,1,0)
        cv.putText(img, f'Count: ', (1000,100), cv.FONT_HERSHEY_PLAIN, 2, (255,255,255),3)
        # initial a detection array
        detections = np.empty((0, 5))
        for r in results:
                boxes = r.boxes
                for box in boxes:
                        # extract bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        # draw bounding box
                        # cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                        # get Confidence
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        # get Class Name
                        cls = int(box.cls[0])
                        currentClass = classNames[cls]
                        # we're only interested in car/truck/bus with probability>0.4
                        # i.e. if it is a motorbike, it will not show the tag
                        if (currentClass=="car" or currentClass=='truck' or currentClass=="bus") and conf>0.4:
                                currentArray = np.array([x1,y1,x2,y2, conf])
                                # x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                                # cx,cy=int(x1+(x2-x1)//2),int(y1+(y2-y1)//2)
                                # cv.circle(img,(cx,cy),5,(255,0,255),cv.FILLED)
                                detections = np.vstack((detections, currentArray)) # append to the list, in np use stack
                                # cv.putText(img, f'{classNames[cls]} {conf}', (max(0, x1), max(20, y1)), cv.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), 2)

        # update the results of tracker        
        resultsTracker = tracker.update(detections)

        # draw a line 
        cv.line(img,(points1[0],points1[1]),(points1[2],points1[3]), (255,0,255),3)
        # draw a line 
        cv.line(img,(points2[0],points2[1]),(points2[2],points2[3]), (255,0,255),3)
        for result in resultsTracker:
                x1,y1,x2,y2,id = result
                print(result)
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                # draw the tracking bounding box  
                cv.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
                cv.putText(img, f'id: {int(id)}', (max(0, x1), max(20, y1-5)), cv.FONT_HERSHEY_TRIPLEX, 0.7, (209, 80, 255), 2)
                # if center cross the line, count the number
                cx,cy=x1+(x2-x1)//2,y1+(y2-y1)//2
                cv.circle(img,(cx,cy),5,(255,0,255),cv.FILLED)
                
                if points1[0] < cx < points1[2] and points1[1]-50 < cy < points1[3]+15:
                        vehicles_enter_time[id] = time.time()
                        cv.line(img, (points1[0],points1[1]), (points1[2], points1[3]), (0,255,0), 3)

                if points2[0] < cx < points2[2] and points2[1]-30 < cy < points2[3]+30:
                        vehicles_leave_time[id] = time.time()
                        countSet.add(id)
                        cv.line(img, (points2[0],points2[1]), (points2[2], points2[3]), (0,255,0), 3)
                        cv.putText(img, f'{ len(countSet)}', (1140,100), cv.FONT_HERSHEY_PLAIN, 2, (255,255,255), 3)
                        if id in vehicles_enter_time:
                            vehicles_elaps_time[id] = vehicles_leave_time[id] - vehicles_enter_time[id] 
                            distance = 22
                            speed_ms = distance/vehicles_elaps_time[id]
                            speed_kh = int(speed_ms*3.6) 
                            cv.putText(img, f'{str(speed_kh)}km/h', (max(0, x1+40), max(20, y1-5)), cv.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
                            speed.append(speed_kh) 
                            # average speed
                        if (len(speed) > 3):
                            sum_speed = int(sum(speed))
                            ave_speed = sum_speed // len(speed)
                            cv.putText(img, f'Average Speed: { str(ave_speed)} ', (940,200), cv.FONT_HERSHEY_PLAIN, 2, (255,255,255), 3)
                            if (ave_speed < 35):
                                cv.putText(img, f' Traffic Heavy!!! ', (940,300), cv.FONT_HERSHEY_PLAIN, 2, (255,255,255), 3)

        # calculate time of each frame
        # fps = 1 / (new_frame_time - prev_frame_time)
        # prev_frame_time = new_frame_time
        # print(fps)
        # show the frames
        cv.imshow('img', img)
        # break the loop by press 'q' 
        if cv.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv.destroyAllWindows()

# python yolo-project-speed.py