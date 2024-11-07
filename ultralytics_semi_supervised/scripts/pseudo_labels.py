# scripts/pseudo_label.py
import os
from ultralytics import YOLO

# Directory for pseudo-labeled data
pseudo_label_dir = '/home/akash/ws/dataset/hand_written/test_data/hindi_human_annotation/images'
model_dir = "/home/akash/ws/artifacts/HW/hindi_finetune_250924/HW_hindi_finetune_250924_/weights/best.pt"
# Load the trained model
model = YOLO(model_dir)



# Ensure the directory exists
# os.makedirs(pseudo_label_dir, exist_ok=True)

# Pseudo-label the unlabeled dataset
results = model.predict(source=pseudo_label_dir, save_txt=True, save_conf=False,
                        imgsz = 1024, name = "./hindi", conf = 0.5, save = True, device = "0", iou=0.99)

# The predictions will be saved in YOLO format in the specified directory
