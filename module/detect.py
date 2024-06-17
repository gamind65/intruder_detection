import cv2
import numpy as np
from ultralytics import YOLO
import time

# find if a point is inside a polygon
def isInside(contour, corr):
    return True if cv2.pointPolygonTest(contour, corr, False) == 1 else False

class YoloDetect():
    def __init__(self, model_path, poly, conf_thresh=0.5):
        """
        Parameters:
        model_path: path to YOLO-v8 model
        poly: points' corrdinate of polygon 
        conf_thresh: confident threshold for model prediction
        cam_idx: index of webcam 
        """
        self.model = YOLO(model_path) # load model
        self.poly = np.array(poly, np.int32).reshape(-1,1,2) # points' corrdinate of polygon
        self.conf_threshold = conf_thresh # confidence threshold
        self.count = 0 # count number of people inside the zone 
        self.classes = 0 # class index of 'people' for YOLO
        self.prev_time = time.time() #
        
    # detect people using YOLO-v8
    def predict(self, img):
        return self.model.predict(img, classes=self.classes, conf=self.conf_threshold)
    
    # give warning with given condition
    def detect_warn(self, curr):
        if curr == 0:
            if self.count != curr:
                print("Only person left!")
        else:
            if self.count > curr:
                print("New person in!")
            elif self.count < curr:
                print("One person out!")

    # run function
    def detect(self, frame):
        # read time and show fps
        curr_time = time.time()
        fps = str(int(1/(curr_time-self.prev_time)))
        cv2.putText(frame, f'fps: {fps}', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        self.prev_time = curr_time
        
        results = self.predict(frame)
        
        # draw polygon
        frame = cv2.polylines(frame, [self.poly], True, (0,255,0), 3) # img, [poly], isClosed, color, thickness
        
        current_count = 0 # count number of poeple currently inside of polygon  
        for result in results:
            # iterates through boxes
            for box in result.boxes:
                # get corrdinates of the box
                xA, yA, xB, yB =map(int, box.xyxy[0]) # (left, top, right, bottom) 
                
                # calculate the coordinates of center bottom of the box 
                corr = (int((xA+xB)/2), yB) 
                # cv2.circle(frame, corr, radius=5, color=(255,0,0), thickness=3) # draw the point
                
                # change color of bounding box to red if the person detected is inside and raise current count
                if isInside(self.poly, corr): 
                    color = (0,0,255)
                    current_count += 1
                # else set default bounding box color to blue
                else: color = (255,0,0) 
                
                # draw bounding box
                cv2.rectangle(frame, (xA,yA), (xB,yB), color, 3)
            
        # give warning message and update counter
        self.detect_warn(current_count)
        self.count = current_count
        cv2.putText(frame, f'count: {self.count}', (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        # draw frame
        img = np.array(frame)
        cv2.namedWindow('Intruder Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Intruder Detection', 960, 590) 
        cv2.imshow('Intruder Detection', img)
        
    def track(self, frame):
        # read time and show fps
        curr_time = time.time()
        fps = str(int(1/(curr_time-self.prev_time)))
        cv2.putText(frame, f'fps: {fps}', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        self.prev_time = curr_time
        
        results = self.model.track(frame, conf=0.5, iou=0.5, classes=self.classes)
        
        # draw polygon
        frame = cv2.polylines(frame, [self.poly], True, (0,255,0), 3) # img, [poly], isClosed, color, thickness
        
        current_count = 0 # count number of poeple currently inside of polygon  
        for result in results:
            # iterates through boxes
            for box in result.boxes:
                # get corrdinates of the box
                xA, yA, xB, yB =map(int, box.xyxy[0]) # (left, top, right, bottom) 
                
                # calculate the coordinates of center bottom of the box 
                corr = (int((xA+xB)/2), yB) 
                # cv2.circle(frame, corr, radius=5, color=(255,0,0), thickness=3) # draw the point
                
                # change color of bounding box to red if the person detected is inside and raise current count
                if isInside(self.poly, corr): 
                    color = (0,0,255)
                    current_count += 1
                # else set default bounding box color to blue
                else: color = (255,0,0) 
                
                # draw bounding box
                cv2.rectangle(frame, (xA,yA), (xB,yB), color, 3)
            
        # give warning message and update counter
        self.detect_warn(current_count)
        self.count = current_count
        cv2.putText(frame, f'count: {self.count}', (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
        # draw frame
        img = np.array(frame)
        cv2.namedWindow('Intruder Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Intruder Detection', 960, 590) 
        cv2.imshow('Intruder Detection', img)
    