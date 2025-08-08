from ultralytics import YOLO

# --- 請確認以下路徑是正確的 ---
model_path = 'best.onnx'
data_yaml_path = 'dataset/data.yaml'
# -----------------------------

print(f"--- 開始最小化測試 ---")
print(f"正在載入模型: {model_path}")
print(f"使用資料集: {data_yaml_path}")

try:
    # 載入模型，並強制指定 task
    model = YOLO(model_path, task='detect')

    print("模型載入成功，準備開始驗證...")

    # 執行驗證
    metrics = model.val(data=data_yaml_path,
                        imgsz=640,
                        conf=0.25,  # 使用一個固定的 conf 值
                        iou=0.7,   # 使用一個固定的 iou 值
                        dnn=True,
                        device="0")

    print("--- 驗證成功 ---")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")

except Exception as e:
    print(f"\n--- 發生錯誤 ---")
    # 印出詳細的錯誤追蹤訊息
    import traceback
    traceback.print_exc()