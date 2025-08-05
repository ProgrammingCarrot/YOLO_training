from ultralytics import YOLO

model = YOLO("best.pt")
print(model.names)

model.export(format="onnx",imgsz = 640)

onnx_model = YOLO("best.onnx")