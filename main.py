from ultralytics import YOLO

path = "rack_dataset/data.yaml"

def train(dataset):
    model = YOLO("yolo11n.yaml")
    model = YOLO("yolo11n.pt")
    model = YOLO("yolo11n.yaml").load("yolo11n.pt")

    results = model.train(data=dataset,epochs=100,imgsz=640,resume=True,device = 0,box = 7,dfl = 2)



if __name__ == "__main__":
    train(path)
