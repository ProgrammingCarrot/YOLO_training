from ray import tune
from ultralytics import YOLO
import pandas as pd
import openpyxl
import re
from datetime import datetime
import os, logging
"""""
使用fitness:[P:0,R:0.2,mAP50:0.2,mAP50-95:0.6]
預設:[P:0,R:0,mAP50:0.1,mAP50-95:0.9]
設定腳本:C:/Users/User/dev/YOLO_project/.venv/Lib/site-packages/ultralytics/utils/metrics.py
"""""

model = YOLO("yolo11n.pt")

search_space = {
    "lr0": (1e-5, 1e-2),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    "lrf": (0.0001, 0.1),  # final OneCycleLR learning rate (lr0 * lrf)
    "momentum": (0.9,0.999),  # SGD momentum/Adam beta1
}

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logging.basicConfig(filename=f'log/{date_time}.log', 
        format='%(asctime)s %(levelname)s:%(message)s', datefmt='%I:%M:%S', 
        encoding='utf-8', level=logging.DEBUG)
    
    path = "dataset/data.yaml"
    choice = input("選擇使用的hyperparameter tuning 方法：[1]Genetic Algorithm, [2]Ray tune：")
    if int(choice) == 1:
        model.tune(
            data=path,
            epochs=100,
            iterations=100,
            optimizer="AdamW",
            space=search_space,
            name="AdamW",
            plots=False,
            save=False,
            val=False
        )
    elif int(choice) == 2:
        print("Ray tune mode")
        logging.info("Ray tune mode")
        search_space = {
            "lr0": tune.uniform(1e-4,1e-2),
            "lrf": tune.uniform(1e-3,1e-1),
            "momentum": tune.uniform(0.900,0.999)
        }
        
        results = model.tune(
                data="dataset/data.yaml",
                use_ray=True,
                space=search_space,
                gpu_per_trial=1,
                resume=True,
                val=False,
                plots=False,
                save=False,
                optimizer="AdamW"
            )
        
        if results.errors:
            print("One or more trials failed!")
            logging.error("One or more trials failed!")
        else:
            print("No errors!")
            logging.info("No errors!")
        
        try:
            log_data = []
            trails_result = []
            for i,result in enumerate(results):
                tune_result = f"Trial #{i}: Configuration: {result.config}, Last Reported Metrics: {result.metrics}"
                log_data.append(tune_result)
                print(tune_result)
                logging.info(tune_result)
            
            for i,data in enumerate(log_data):
                if not data:
                    continue
                # 建立一個 dictionary 來存放單一 trial 的數據
                trial_data = {}
                
                # 從文字中提取各種數值
                # 使用正規表示式 (regular expression) 來尋找並抓取數字
                trial_data['Trial'] = int(re.search(r'(\d+):', data).group(1))
                trial_data['lr0'] = float(re.search(r"'lr0': np\.float64\((.*?)\)", data).group(1))
                trial_data['lrf'] = float(re.search(r"'lrf': np\.float64\((.*?)\)", data).group(1))
                trial_data['momentum'] = float(re.search(r"'momentum': np\.float64\((.*?)\)", data).group(1))
                
                trial_data['Precision'] = float(re.search(r"'metrics/precision\(B\)': ([\d\.]+)", data).group(1))
                trial_data['Recall'] = float(re.search(r"'metrics/recall\(B\)': ([\d\.]+)", data).group(1))
                trial_data['mAP50'] = float(re.search(r"'metrics/mAP50\(B\)': ([\d\.]+)", data).group(1))
                trial_data['mAP50-95'] = float(re.search(r"'metrics/mAP50-95\(B\)': ([\d\.]+)", data).group(1))
                
                trial_data['val/box_loss'] = float(re.search(r"'val/box_loss': ([\d\.]+)", data).group(1))
                trial_data['val/cls_loss'] = float(re.search(r"'val/cls_loss': ([\d\.]+)", data).group(1))
                trial_data['val/dfl_loss'] = float(re.search(r"'val/dfl_loss': ([\d\.]+)", data).group(1))
                
                trails_result.append(trial_data)

            # 3. 建立 DataFrame
            df = pd.DataFrame(trails_result)

            # 4. 儲存為 Excel 檔案
            output_filename = 'training_results.xlsx'
            df.to_excel(output_filename, index=False, engine='openpyxl')

            print(f"Data Record Success: {output_filename}")
            logging.critical(f"Data Record Success: {output_filename}")
        except Exception as e:
            print(f"Error! {e}")
            logging.error(f"Error! {e}")