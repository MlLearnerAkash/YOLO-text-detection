import os
import cv2
import numpy as np

# Convert YOLO normalized coordinates to pixel coordinates
def yolo_to_bbox(image_shape, bbox):
    img_h, img_w = image_shape[:2]
    x_center, y_center, width, height = bbox
    x_center, y_center, width, height = x_center * img_w, y_center * img_h, width * img_w, height * img_h
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    return x1, y1, x2, y2

# Function to draw boxes on the image
def draw_boxes(image, boxes, color, label):
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Read annotations from text file (YOLO format: class_id, x_center, y_center, width, height)
def read_annotations(file_path):
    boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:  # Ensure valid format
                _, x_center, y_center, width, height = map(float, parts)
                boxes.append((x_center, y_center, width, height))
    return boxes

# Main function to process directories
def process_images(gt_dir, pred_dir, img_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image {img_path}. Skipping...")
            continue
        
        base_name = os.path.splitext(img_file)[0]
        gt_file = os.path.join(gt_dir, f"{base_name}.txt")
        pred_file = os.path.join(pred_dir, f"{base_name}.txt")
        
        # Read ground truth and predictions
        if os.path.exists(gt_file):
            gt_boxes = read_annotations(gt_file)
            gt_boxes = [yolo_to_bbox(img.shape, bbox) for bbox in gt_boxes]
            draw_boxes(img, gt_boxes, (0, 255, 0), "GT")  # Green for Ground Truth

        if os.path.exists(pred_file):
            pred_boxes = read_annotations(pred_file)
            pred_boxes = [yolo_to_bbox(img.shape, bbox) for bbox in pred_boxes]
            draw_boxes(img, pred_boxes, (0, 0, 255), "Pred")  # Red for Predictions
        
        # Save the image with both GT and Predictions drawn
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, img)
        print(f"Saved result to {output_path}")

# Example usage
gt_dir = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/test_data/gt'
pred_dir = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/test_data/labels'
img_dir = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/test_data/'
output_dir = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/test_data/'

process_images(gt_dir, pred_dir, img_dir, output_dir)
