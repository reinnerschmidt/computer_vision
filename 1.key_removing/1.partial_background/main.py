import cv2
import pandas as pd
import numpy as np
import argparse

#define arguments
parser=argparse.ArgumentParser()
parser.add_argument('-i', '--video_input_path', required=True, help='Path of input video')
parser.add_argument('-b', '--video_background_path', required=True, help='Path of background video')
parser.add_argument('-o', '--output', required=True, help='Nome do arquivo de saída')
args=parser.parse_args()

#First import the video

cap_video= cv2.VideoCapture(args.video_input_path)
cap_background = cv2.VideoCapture(args.video_background_path)

#Save codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#Save fps from input
fps = cap_video.get(cv2.CAP_PROP_FPS)

#Save resolution
frame_width = int(cap_video.get(3))
frame_height = int(cap_video.get(4))

#Save video using CLI
out=cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height) )

partial_background = cv2.VideoWriter('partial_background.mp4', fourcc,fps, (frame_width, frame_height) )
partial_input = cv2.VideoWriter('partial_input.mp4', fourcc,fps, (frame_width, frame_height) )


#Main Loop
while True:
    #read videos
    
    #ret_video = check if video still playing
    ret_video,frame_video = cap_video.read() 
    ret_background,frame_background = cap_background.read()
    
    #check if video ended
    if not ret_video or not ret_background:
        break

    #Define upper and lower limites to green in R
    #create lower limit to green array as uint8
    lower_green = np.array([0,100,0],dtype=np.uint8) 
    
    #create upper limit to green array as uint8
    upper_green = np.array([100,255,100],dtype=np.uint8) 

    #creat green mask

    #use frame of video as base to create mask with lower and upper green
    green_mask = cv2.inRange(frame_video,lower_green,upper_green) 

    #remove the parts of background that is in the green_mask
    filter_background = cv2.bitwise_and(frame_background,frame_background,mask=green_mask)  

    #Create a file that is just in the green background 
    partial_background.write(filter_background)

    #remove the parts of video input that aren´t in the green_mask
    #to do that we need to create an inverse green_mask
    inv_green_mask = np.invert(green_mask)

    filter_video = cv2.bitwise_and(frame_video,frame_video,mask=inv_green_mask)

    #Create a file that is just in the green background 
    partial_input.write(filter_video)



    #merge the filtered videos using weighted  method
    result = cv2.addWeighted(filter_background,1,filter_video,1,0)

    #Save result in out
    out.write(result)

    #Show video
    cv2.imshow("Resultado",result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_video.release()
cap_background.release()
cv2.destroyAllWindows()



    

    






