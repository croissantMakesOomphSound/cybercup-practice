from untitled0 import ObjectDetection,AnnotatedVideoPlayer, iotsensorvalue,gpsvalue,mobileapps, traffic

object1=iotsensorvalue()
iotSensorEntry=object1.iotentry()
iotSensorExit=object1.iotexit()
iot_net = (iotSensorExit - iotSensorEntry) if (iotSensorEntry is not None and iotSensorExit is not None) else 0

object2=gpsvalue()
gpsvalue=object2.getgpsvalue()


object3=mobileapps()
mobilevalue=object3.getmobilevalue()

object4=traffic()
redLight=object4.getredLight()
yellowLight=object4.getyellowlight()
greenLight=object4.getgreenLight()


# Open the video file
input_video_path = r"C:\Users\TMpub\OneDrive\Desktop\Recording 2024-10-17 214815.mp4"  # Your video file path
output_video_path = 'annotated_output4.mp4'  # Path to save the annotated video

object5=ObjectDetection(input_video_path,output_video_path)
object5.run_model()
count=object5.get_count()
print("count"+str(count))

player = AnnotatedVideoPlayer(output_video_path)
player.play_video()

current_traffic=max(iot_net,gpsvalue,mobilevalue,count)
