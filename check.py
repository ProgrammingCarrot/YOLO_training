import os

# --- 請修改此處的路徑 ---
# 指向您「驗證集」的標籤資料夾 (e.g., 'dataset/labels/val')
path_to_labels = 'dataset/valid' 
# -------------------------

problematic_files = []

print(f"開始掃描資料夾：{path_to_labels}")

if not os.path.isdir(path_to_labels):
    print(f"錯誤：找不到資料夾 {path_to_labels}，請確認路徑是否正確。")
else:
    for filename in os.listdir(path_to_labels):
        if filename.endswith('.txt'):
            filepath = os.path.join(path_to_labels, filename)
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    if not lines: # 檢查是否為空檔案
                        problematic_files.append(f"{filename} (檔案為空)")
                        continue

                    for i, line in enumerate(lines):
                        line = line.strip()
                        if not line: # 檢查是否為空白行
                            problematic_files.append(f"{filename} (第 {i+1} 行是空白行)")
                            continue
                        
                        parts = line.split()
                        if len(parts) < 7: # 一個 class_id + 至少3個頂點 (6個座標)
                            problematic_files.append(f"{filename} (第 {i+1} 行座標點不足，總部份數: {len(parts)})")
                        
                        # 檢查座標點是否為偶數
                        if (len(parts) - 1) % 2 != 0:
                            problematic_files.append(f"{filename} (第 {i+1} 行座標點數量不是偶數)")

            except Exception as e:
                problematic_files.append(f"{filename} (讀取時發生錯誤: {e})")

if problematic_files:
    print("\n--- 發現以下有問題的檔案 ---")
    for problem in problematic_files:
        print(problem)
    print("\n請檢查並修正以上檔案後再重新訓練。")
else:
    print("\n恭喜！所有標籤檔案格式初步檢查通過。")