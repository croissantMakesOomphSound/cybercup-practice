import cv2
from ultralytics import YOLO
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import math
import os

class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

class ObjectDetection:
    count = 0  # Class attribute to store the total count of vehicles

    def __init__(self, input_video_path=None, output_video_path=None):
        self.input_video_path = input_video_path 
        self.output_video_path = output_video_path
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++"+str(output_video_path))
        self.tracker = Tracker()
        self.features_list = []
        self.kmeans = KMeans(n_clusters=5)
        self.vehicle_id_counter = 0

        # Load model and feature extractor
        self.model = YOLO('yolov8n.pt')
        self.resnet_model = models.resnet50(pretrained=True)
        self.resnet_model.eval()
        self.feature_extractor = torch.nn.Sequential(*(list(self.resnet_model.children())[:-1]))

        # Preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def run_model(self):
        cap = cv2.VideoCapture(self.input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 0.5)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                results = self.model(frame)
                filtered_results = []
                
                for result in results:
                    for box, score, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                        if score > 0.5:
                            filtered_results.append(box)
                            
            detected_cars = []
            for box in filtered_results:
                x1, y1, x2, y2 = map(int, box)
                cropped_img = frame[y1:y2, x1:x2]
                cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                input_tensor = self.preprocess(cropped_img_pil).unsqueeze(0)
                            
                with torch.no_grad():
                    feature_vector = self.feature_extractor(input_tensor).flatten().numpy()
                    
                    self.features_list.append(feature_vector)
                    
                    vehicle_id = self.tracker.update([(x1, y1, x2 - x1, y2 - y1)])[0][-1]
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"ID: {vehicle_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    detected_cars.append({'id': vehicle_id, 'bbox': (x1, y1, x2, y2)})
                    ObjectDetection.count += 1  # Increment count when a vehicle is detected

            frame_count += 1
            out.write(frame)
            

        cap.release()
        out.release()
        print(f"Annotated video saved to: {self.output_video_path}")

        # Clustering to determine unique vehicles
        if self.features_list:
            self.kmeans.fit(self.features_list)
            unique_vehicle_count = len(set(self.kmeans.labels_))  # Count unique clusters
            print(f"Total Unique Vehicles Detected: {unique_vehicle_count}")
            ObjectDetection.count = unique_vehicle_count  # Update the class count

    @classmethod
    def get_count(cls):
        return cls.count  # Returns the total number of vehicles detected

    

class AnnotatedVideoPlayer:
    def __init__(self, output_video_path):
        self.output_video_path = output_video_path

    def play_video(self):
        cap = cv2.VideoCapture(self.output_video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow('Annotated Video', frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
class iotsensorvalue:
    def __init__(self,iotSensorEntry=None,iotSensorExit=None):
        self.iotSensorEntry=iotSensorEntry
        self.iotSensorExit=iotSensorExit
    def iotentry(self):
        return self.iotSensorEntry
    def iotexit(self):
        return self.iotSensorExit
    
class gpsvalue:
    def __init__(self,gpsvalue=None):
        self.gpsvalue=gpsvalue
    def getgpsvalue(self):
        return self.gpsvalue
    
class mobileapps:
    def __init__(self,mobileapps=None):
        self.mobileapps=mobileapps
    def getmobilevalue(self):
        return self.mobileapps


class traffic:

    def __init__(self,redLight=60,yellowLight=15,greenLight=30):
        self.redLight=redLight
        self.yellowLight=yellowLight
        self.greenLight=greenLight
        
    def getredLight(self):
        return self.redLight
    def getyellowlight(self):
        return self.yellowLight
    def getgreenLight(self):
        return self.greenLight
    
class trafficlightsclock:
    def __init__(self,redclock=None,yellowclock=None,greenclock=None,masterclock=None):
        self.redclock=redclock
        self.yellowclock=yellowclock
        self.greenclock=greenclock
        self.masterclock=masterclock
   
    def get_redclock(self):
        return self.redclock

    def set_redclock(self, redclock):
        self.redclock = redclock

    def get_yellowclock(self):
        return self.yellowclock

    def set_yellowclock(self, yellowclock):
        self.yellowclock = yellowclock

    def get_greenclock(self):
        return self.greenclock

    def set_greenclock(self, greenclock):
        self.greenclock = greenclock

    def get_masterclock(self):
        return self.masterclock

    def set_masterclock(self, masterclock):
        self.masterclock = masterclock

    

    
