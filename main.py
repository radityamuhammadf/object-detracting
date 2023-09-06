# pip install opencv-contrib-python
# pip install ultralytics
# pip install supervision

from ultralytics import YOLO
import supervision as sv    
import cv2

#create a line for counting
#the default resolution for detection is 640x480
res_height=int(480)
# create a start and end of a line coordinate --> used for counting the object 
START=sv.Point(0,res_height//2)
END=sv.Point(640,res_height//2)

def main():
    # initiate model --> using pretrained yolov8
    model=YOLO("yolov8l.pt")
    
    # create a line zone 
    line_zone=sv.LineZone(start=START,end=END)
    line_zone_annotator=sv.LineZoneAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=0.5
    )
    
    # create a bounding box using BoxAnnotator class from Supervision library
    box_annotator=sv.BoxAnnotator(
        thickness=2,
        text_thickness= 1,
        text_scale=0.5
    )
    # iterating each frame that tracked using "model.track" method
    for result in model.track(source="test.mp4",show=False,stream=True):
        # set frame variable using the original image (orig_img attribute) from method output
        frame=result.orig_img
        # call YOLOv8 for image detection
        detections=sv.Detections.from_yolov8(result)
        # create an tracker instance 
        if result.boxes.id is not None:
            detections.tracker_id=result.boxes.id.cpu().numpy().astype(int)
        
        # filter only "person" class that will be detected
        detections=detections[detections.class_id==0]

        #instatiate labels variable
        labels=[
            f"#{tracker_id} - {model.model.names[class_id]} {confidence:0.2f}"
            for _,confidence,class_id,tracker_id
            in detections
        ]
        
        # create box annotator using "frame" variable and detections that initiated in "detections" variables
        frame=box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        # annotate the line zone
        line_zone.trigger(detections=detections)
        line_zone_annotator.annotate(frame=frame,line_counter=line_zone)

        # show the result of the annotated frame
        cv2.imshow("yolov8",frame)
        
        # breaking the "for-loop"/closing the detection if Esc button triggered 
        if(cv2.waitKey(10)==27):
            break


if __name__=="__main__":
    main()