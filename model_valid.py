from ultralytics import YOLO
import sys

if __name__ == "__main__":
    model_name = sys.argv[1]
    conf_value = float(sys.argv[2])
    IOU_value = float(sys.argv[3])
    if (sys.argv[4] == "dnn"):
        dnn = True
    else:
        dnn = False
    model = YOLO(model_name,task="detect")
    #Customize validation settings
    metrics = model.val(data="dataset/data.yaml", 
                        imgsz=640, 
                        conf=conf_value,
                        iou=IOU_value,
                        dnn=dnn,
                        device="0")
    print(metrics.summary())