import cv2
import os

# Threshold for IoU to consider boxes overlapping
iou_threshold = 0.05

def draw_boxes(image, boxes, color):
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

def get_boxes(annotation_file, width, height):
    boxes = []
    with open(annotation_file, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        elements = line.strip().split()
        class_id = int(elements[0])
        x_center = float(elements[1]) * width
        y_center = float(elements[2]) * height
        box_width = float(elements[3]) * width
        box_height = float(elements[4]) * height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        boxes.append([x1, y1, x2, y2])
    
    return boxes

def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate intersection area
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calculate areas of both bounding boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union area
    union_area = area_box1 + area_box2 - inter_area

    # Calculate IoU
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def merge_boxes(box1, box2):
    # Calculate the merged bounding box coordinates
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])

    return [x_min, y_min, x_max, y_max]

def eliminate_overlaps(boxes):
    merged_boxes = []
    
    while boxes:
        current_box = boxes.pop(0)
        to_merge = []
        
        for other_box in boxes:
            if calculate_iou(current_box, other_box) > iou_threshold:
                to_merge.append(other_box)

        # Merge overlapping boxes
        for box_to_merge in to_merge:
            current_box = merge_boxes(current_box, box_to_merge)
            boxes.remove(box_to_merge)
        
        merged_boxes.append(current_box)
    
    return merged_boxes

def annotate_images(image_dir, annotation_files1, annotation_files2, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = sorted(os.listdir(image_dir))

    for image_name in images:
        if not image_name.endswith('.jpg') and not image_name.endswith('.png'):
            continue
        
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        annotation_file1 = os.path.join(annotation_files1, f"{os.path.splitext(image_name)[0]}.txt")
        annotation_file2 = os.path.join(annotation_files2, f"{os.path.splitext(image_name)[0]}.txt")

        if os.path.exists(annotation_file1):
            boxes1 = get_boxes(annotation_file1, width, height)
        else:
            boxes1 = []

        if os.path.exists(annotation_file2):
            boxes2 = get_boxes(annotation_file2, width, height)
        else:
            boxes2 = []

        # Combine all boxes from both annotations
        all_boxes = boxes1 + boxes2

        # Eliminate overlapping boxes and create a single bounding box for each detection
        final_boxes = eliminate_overlaps(all_boxes)

        # Draw final merged boxes
        draw_boxes(image, final_boxes, (0, 255, 0))  # Green for final bounding boxes
        # draw_boxes(image, boxes1, (255, 0, 0))  # Green for final bounding boxes

        output_image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_image_path, image)

if __name__ == "__main__":
    image_directory = '/home/akash/ws/dataset/hand_written/hindi_finetune_batch1/ravi_style_gt/labels/openseg_annot/images'
    annotation_directory2 = '/home/akash/ws/dataset/hand_written/hindi_finetune_batch1/ravi_style_gt/labels/openseg_annot/annotations'
    annotation_directory1 = '/home/akash/ws/YOLO-text-detection/ultralytics/runs/detect/predict16/labels'
    output_directory = './output'

    annotate_images(image_directory, annotation_directory1, annotation_directory2, output_directory)
