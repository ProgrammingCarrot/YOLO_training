from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("best.pt")
    print(model.names)

    model.export(format="onnx",
                 imgsz=640,
                 dnn=True,
                 device="0",
                 task="detect")

    onnx_model = YOLO("best.onnx")