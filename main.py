# pip install opencv-contrib-python
# pip install ultralytics
# pip install supervision

from ultralytics import YOLO
import supervision as sv    
import cv2

def main():
    # initiate model --> using pretrained yolov8
    model=YOLO("yolov8l.pt")
    # iterating each frame that tracked using "model.track" method
    for result in model.track(source=0,show=False,stream=True):
        frame=result.orig_img
        
        cv2.imshow("yolov8",frame)

        # breaking the "for-loop"/closing the detection if Esc button triggered 
        if(cv2.waitKey(30)==27):
            break


if __name__=="__main__":
    main()