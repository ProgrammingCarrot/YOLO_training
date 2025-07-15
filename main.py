from ultralytics import YOLO

def train():
    model = YOLO("yolo11n.yaml")
    model = YOLO("yolo11n.pt")
    model = YOLO("yolo11n.yaml").load("yolo11n.pt")

    results = model.train(data="dataset/data.yaml",epochs=100,imgsz=640,resume=True)



if __name__ == "__main__":
    train()
