from ultralytics import YOLO
import sys

# Load a model
model = YOLO("best.pt")
source = "test_media/test.jpg"

if __name__ == "__main__":
    conf_value = float(sys.argv[1])
    IOU_value = float(sys.argv[2])
    # Customize validation settings
    metrics = model.val(data="rack_dataset/data.yaml", 
                        imgsz=640, 
                        conf=conf_value,
                        iou=IOU_value, 
                        device="0")
    result = model.predict(source, imgsz=640, stream=True, save=True)
    print(metrics.summary())