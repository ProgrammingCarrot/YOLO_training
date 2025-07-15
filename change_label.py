import os
import re

# --- 請修改此處的路徑 ---
# 指向您「驗證集」的標籤資料夾 (e.g., 'dataset/labels/val')
path_to_labels = 'dataset/valid/labels' 
# -----------------------

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

                    for i, line in enumerate(lines):
                        
                        parts = line.split()
                        if int(parts[0]) == 1:
                            problematic_files.append(f"{filename}")

            except Exception as e:
                problematic_files.append(f"{filename} (讀取時發生錯誤: {e})")

if problematic_files:
    print("\n--- 發現以下有問題的檔案 ---")
    for files in problematic_files:
        print(f'正在重寫:{files}')
        path = f'{path_to_labels}/{files}'
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 2. 在記憶體的字串變數中進行替換
        new_content = re.sub(r'\b1\b','0',content)
        print("內容已在記憶體中替換。")

        # 3. 再用寫入模式 ('w') 打開同一個檔案，將修改後的完整內容一次性寫回
        #    'w' 模式會自動清空原檔案，正好是我們需要的
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            print("新內容已成功寫回檔案！")

    print("\n所有檔案更新完畢。")
else:
    print("\n恭喜！所有標籤檔案格式初步檢查通過。")