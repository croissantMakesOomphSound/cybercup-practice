from untitled0 import ObjectDetection,AnnotatedVideoPlayer, iotsensorvalue,gpsvalue,mobileapps, traffic
from untitled3 import log_car_count
import random

object1=iotsensorvalue(random.randint(0,10))
iotSensorEntry=object1.iotentry(random.randint(0,10))
iotSensorExit=object1.iotexit(random.randint(0,10))
iot_net = (iotSensorExit - iotSensorEntry) if (iotSensorEntry is not None and iotSensorExit is not None) else 0

object2=gpsvalue(random.randint(0,10))
gpsvalue=(object2.getgpsvalue()) if (object2.getgpsvalue()!=None) else 0


object3=mobileapps(random.randint(0,10))
mobilevalue=(object3.getmobilevalue()) if (object3.getmobilevalue()!=None) else 0

object4=traffic()
redLight=object4.getredLight()
yellowLight=object4.getyellowlight()
greenLight=object4.getgreenLight()


# Open the video file
input_video_path = r"+C:\Users\TMpub\OneDrive\Desktop\Recording 2024-10-17 002217.mp4"  # Your video file path
output_video_path = 'annotated_output5.mp4'  # Path to save the annotated video

object5=ObjectDetection(input_video_path,output_video_path)
object5.run_model()
count=object5.get_count()
print("count"+str(count))

current_traffic=max(iot_net,gpsvalue,mobilevalue,count)
log_car_count(current_traffic)
player = AnnotatedVideoPlayer(output_video_path)
player.play_video()



