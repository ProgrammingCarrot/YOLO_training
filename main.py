from ultralytics import YOLO



def train(dataset):
    model = YOLO("yolo11s.yaml")
    model = YOLO("yolo11s.pt")
    model = YOLO("yolo11s.yaml").load("yolo11s.pt")
    results = model.train(
        data=dataset,
        epochs=100,
        imgsz=640,
        resume=True,
        save = True,
        device = 0,
        patience = 10)    
    
if __name__ == "__main__":
    path = "rack_dataset/data.yaml"
    train(path)
