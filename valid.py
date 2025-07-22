from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")

if __name__ == "__main__":
    # Customize validation settings
    metrics = model.val(data="rack_dataset/data.yaml", imgsz=640, conf=0.25, device="0")
    print(metrics.summary())