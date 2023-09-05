# pip install opencv-contrib-python
# pip install ultralytics
# pip install supervision

from ultralytics import YOLO
import supervision as sv    
import cv2

def main():
    # initiate model --> using pretrained yolov8
    model=YOLO("yolov8l.pt")
    # create a bounding box using BoxAnnotator class from Supervision library
    box_annotator=sv.BoxAnnotator(
        thickness=2,
        text_thickness= 1,
        text_scale=0.5
    )
    # iterating each frame that tracked using "model.track" method
    for result in model.track(source=0,show=False,stream=True):
        # set frame variable using the original image (orig_img attribute) from method output
        frame=result.orig_img
        # call YOLOv8 for image detection
        detections=sv.Detections.from_yolov8(result)
        # create box annotator using "frame" variable and detections that initiated in "detections" variables
        frame=box_annotator.annotate(scene=frame,detections=detections)
        cv2.imshow("yolov8",frame)
        
        # breaking the "for-loop"/closing the detection if Esc button triggered 
        if(cv2.waitKey(30)==27):
            break


if __name__=="__main__":
    main()