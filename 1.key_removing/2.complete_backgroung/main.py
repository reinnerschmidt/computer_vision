import cv2
import pandas as pd
import numpy as np
import argparse

#Define arguments for CLI inputs
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_path', required=True, help='input Video path')
parser.add_argument('-o', '--output_name', required=True, help='Name of the output result video')
args=parser.parse_args()

#import videos
cap_video = cv2.VideoCapture(args.input_path)


#Codec para mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#get resolution
frame_width = int(cap_video.get(3))
frame_height = int(cap_video.get(4))

#fps
fps = cap_video.get(5)

#create output video file
output = cv2.VideoWriter(args.output_name, fourcc, fps,(frame_width, frame_height))

#main loop
while True:
    
    #read framne
    ret_cap_video, frame_cap_video = cap_video.read()
    

     #check if video ended
    if not ret_cap_video: 
        break
    
    #Define upper and lower limites to green 
    #create lower limit to green array as uint8
    lower_green = np.array([30,150,20],dtype=np.uint8) 
    
    #create upper limit to green array as uint8
    upper_green = np.array([85,255,255],dtype=np.uint8) 

    #create HSV frame
    frame_cap_video_hsv = cv2.cvtColor(frame_cap_video,cv2.COLOR_BGR2HSV)

    #creat green mask
    #use HSV frame of input video as base to create mask with lower and upper green
    green_mask = cv2.inRange(frame_cap_video_hsv,lower_green,upper_green) 

    #remove the parts of  input video that are in the green_mask
    #to do that we need to create an inverse green_mask
    inv_green_mask = cv2.bitwise_not(green_mask)

    #create masked video
    filter_video = cv2.bitwise_and(frame_cap_video,frame_cap_video,mask=inv_green_mask)

    #Create a file that aren't in the green area
    output.write(filter_video)

    #Show video
    cv2.imshow("Resultado",filter_video)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap_video.release()
cap_background.release()
cv2.destroyAllWindows()

    
    




