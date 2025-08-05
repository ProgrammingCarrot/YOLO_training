from ultralytics import YOLO
import sys

def train(dataset,lr0,lrf,momentum):
    model = YOLO("yolo11s.yaml").load("yolo11s.pt")
    results = model.train(
        data=dataset,
        epochs=100,
        imgsz=640,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        dropout=0.0,
        optimizer="AdamW",
        resume=False,
        save = True,
        device = 0,
        patience = 100)    
    
if __name__ == "__main__":
    path = "dataset/data.yaml"
    lr0_value = sys.argv[1]
    lrf_value = sys.argv[2]
    momentum = sys.argv[3]
    train(path,lr0_value,lrf_value,momentum)
