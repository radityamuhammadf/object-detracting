# pip install opencv-contrib-python
# pip install ultralytics
# pip install supervision

from ultralytics import YOLO
import supervision as sv    

def main():
    model=YOLO("yolov8l.pt")
    result=model.track(source=0,show=True)

if __name__=="__main__":
    main()