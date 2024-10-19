import cv2
from untitled0 import ObjectDetection, AnnotatedVideoPlayer, iotsensorvalue, gpsvalue, mobileapps, traffic
from untitled3 import log_car_count, trainmodel
import random
from moviepy.video.io.VideoFileClip import VideoFileClip
import os

congestion=0
object0=traffic()
greenLight=object0.getgreenLight()
yellowLight=object0.getyellowlight()
redLight=object0.getredLight()
# Initialize sensor values
object1 = iotsensorvalue(random.randint(0, 5), random.randint(0, 10))
iotSensorEntry = object1.iotentry()
iotSensorExit = object1.iotexit()
iot_net = (iotSensorExit - iotSensorEntry) if (iotSensorEntry is not None and iotSensorExit is not None) else 0

object2 = gpsvalue(random.randint(0, 5))
gps_value = (object2.getgpsvalue()) if (object2.getgpsvalue() is not None) else 0

object3 = mobileapps(random.randint(0, 5))
mobile_value = (object3.getmobilevalue()) if (object3.getmobilevalue() is not None) else 0

object4 = traffic()
redLight = object4.getredLight()
yellowLight = object4.getyellowlight()
greenLight = object4.getgreenLight()

# Open the video file
input_video_path = r"C:\Users\TMpub\OneDrive\Desktop\Recording 2024-10-17 214815.mp4"  # Your video file path
output_video_path = 'annotated_output8.mp4'  # Path to save the annotated video
temp_path='temp.mp4'
temp_output_path=None
video = VideoFileClip(input_video_path)


output_folder = r"C:\Users\TMpub\cybercup4\annotated_output7"
# Create the folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print("folder created")
    
else:
    print("folder exists")


    
# Get the duration of the video
duration = int(video.duration)  # in seconds
i = 0

# Process every 5 seconds
for start_time in range(0, duration, 5):
    end_time = min(start_time + 5, duration)
    segment = video.subclip(start_time, end_time)
    segment.write_videofile(temp_path, codec='libx264')
    
    

    # Construct the path for the output file
    temp_output_path = output_folder+ r'\annotated_output_'+f'{i}'+'.mp4'
    
    
    # Initialize ObjectDetection outside of the loop
    object5 = ObjectDetection(temp_path, temp_output_path)
    
    # object5 = ObjectDetection(temp_path, temp_Output_path)
    # Process the segment
    print(f"Processing segment from {start_time} to {end_time}")
    
    object5.run_model()  # Run detection on the segment
    count = object5.get_count()
    print("Count: " + str(count))

    current_traffic = max(iot_net, gps_value, mobile_value, count)
    log_car_count(current_traffic)
    
    player = AnnotatedVideoPlayer(output_video_path)
    player.play_video()

    
    print("################################" + str(i + 1))
    i += 1

# After processing all segments, train the model (if needed)
result=trainmodel()
actual=result[0]
predicted=result[1]

if(actual>predicted):
    congestion=1+(actual-predicted)/predicted
elif(actual=predicted):
    congestion=1
else:
    congestion=0
    
    
if(congestion>1):
    redLight+= (redlight*congestion) if(redlight*congestion>5) else 5
    

# Optionally, play the annotated video after processing
player = AnnotatedVideoPlayer(output_video_path)
player.play_video()

# Close the video file
video.close()

