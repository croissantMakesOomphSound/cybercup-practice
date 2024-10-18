import cv2
from ultralytics import YOLO
import torch
from torchvision import models, transforms
from PIL import Image  # Import PIL for image conversion

class ObjectDetection:
    count=0
    def __init__(self, input_video_path=None, output_video_path=None):
        self.input_video_path = input_video_path if input_video_path else r'C:\Users\TMpub\OneDrive\Desktop\Recording 2024-10-17 214815.mp4'
        self.output_video_path = output_video_path if output_video_path else 'annotated_output4.mp4'
        self.vehicle_id_counter = 0
        self.tracked_vehicles = {}
        self.unique_features = set()

    def run_model(self):
        model = YOLO('yolov8n.pt')  
        resnet_model = models.resnet50(pretrained=True)
        resnet_model.eval()
        feature_extractor = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        cap = cv2.VideoCapture(self.input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        desired_classes = {0, 2, 3, 5, 7}
        confidence_threshold = 0.5

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            filtered_results = []
            for result in results:
                for box, score, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    if int(class_id) in desired_classes and score > confidence_threshold:
                        filtered_results.append((box, score, class_id))

            detected_cars = []
            for box, score, class_id in filtered_results:
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(class_id)]}: {score:.2f}"

                # Extract features for tracking
                cropped_img = frame[y1:y2, x1:x2]
                
                # Convert to PIL Image
                cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

                # Preprocess the image
                input_tensor = preprocess(cropped_img_pil).unsqueeze(0)  # Add batch dimension
                
                with torch.no_grad():
                    feature_vector = feature_extractor(input_tensor).flatten()

                # Extract features for tracking
                cropped_img = frame[y1:y2, x1:x2]
                cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                input_tensor = preprocess(cropped_img_pil).unsqueeze(0)

                with torch.no_grad():
                    feature_vector = feature_extractor(input_tensor).flatten().numpy()  # Convert to NumPy array

                # Check for uniqueness and update the count
                feature_tuple = tuple(feature_vector)  # Convert to tuple for set operations
                if feature_tuple not in self.unique_features:
                    self.unique_features.add(feature_tuple)
                    ObjectDetection.count += 1  # Increment the unique count
                # Track vehicles
                vehicle_id = self.track_vehicle((x1, y1, x2, y2), feature_vector)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {vehicle_id} {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                detected_cars.append({'id': vehicle_id, 'bbox': (x1, y1, x2, y2)})

            # Update vehicle positions and count unique vehicles
            self.update_tracked_vehicles(detected_cars)

            out.write(frame)

        cap.release()
        out.release()
        print(f"Annotated video saved to: {self.output_video_path}")

    def track_vehicle(self, bbox, feature_vector):
        vehicle_id = self.vehicle_id_counter
        self.vehicle_id_counter += 1
        self.tracked_vehicles[vehicle_id] = {'bbox': bbox, 'features': feature_vector}
        return vehicle_id

    def update_tracked_vehicles(self, detected_cars):
        for car in detected_cars:
            car_id = car['id']
            if car_id in self.tracked_vehicles:
                self.tracked_vehicles[car_id]['bbox'] = car['bbox']

    @classmethod
    def get_count(cls):
        return cls.count


   

    def extract_features(self, detected_cars, frame):
        features = []
        
        for car in detected_cars:
            # Get bounding box for the detected car
            x1, y1, x2, y2 = car['bbox']  # Example format: [x1, y1, x2, y2]
            
            # Crop the detected car image
            cropped_img = frame[y1:y2, x1:x2]
            
            # Convert to PIL Image
            cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            
            # Preprocess the image
            input_tensor = preprocess(cropped_img_pil).unsqueeze(0)  # Add batch dimension
            
            # Extract features
            with torch.no_grad():
                feature_vector = feature_extractor(input_tensor)
                features.append(feature_vector.flatten())  # Flatten to 1D array
                
                return features


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

    

    
