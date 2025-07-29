from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")
source = "test_media/test.jpg"

if __name__ == "__main__":
    # Customize validation settings
    metrics = model.val(data="rack_dataset/data.yaml", imgsz=640, conf=0.7, device="0")
    result = model.predict(source, imgsz=640, stream=True, save=True)
    print(metrics.summary())