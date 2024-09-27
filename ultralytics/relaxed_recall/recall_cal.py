# import numpy as np
# import torch
# from scipy.spatial import ConvexHull
# from typing import List, Tuple, Dict

# class BoundingBoxEvaluator:
#     def __init__(self, image_width: int, image_height: int, iou_threshold: float = 0.5):
#         self.image_width = image_width
#         self.image_height = image_height
#         self.iou_threshold = iou_threshold

#     def read_boxes_from_file(self, file_path: str) -> List[List[int]]:
#         boxes = []
#         with open(file_path, 'r') as file:
#             for line in file:
#                 box = list(map(float, line.strip().split(' ')))[1:]  # YOLO format: ignoring class id
#                 box = self.denormalize_yolo_bbox(box)
#                 boxes.append(box)
#         return boxes

#     def denormalize_yolo_bbox(self, yolo_bbox: List[float]) -> Tuple[int, int, int, int]:
#         x_center, y_center, width, height = yolo_bbox
#         x_center_pixel = x_center * self.image_width
#         y_center_pixel = y_center * self.image_height
#         width_pixel = width * self.image_width
#         height_pixel = height * self.image_height

#         x_min = int(x_center_pixel - (width_pixel / 2))
#         y_min = int(y_center_pixel - (height_pixel / 2))
#         x_max = int(x_center_pixel + (width_pixel / 2))
#         y_max = int(y_center_pixel + (height_pixel / 2))

#         return x_min, y_min, x_max, y_max

#     def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
#         """Calculate Intersection over Union (IoU) between two bounding boxes."""
#         x1_min, y1_min, x1_max, y1_max = box1
#         x2_min, y2_min, x2_max, y2_max = box2

#         # Determine the coordinates of the intersection rectangle
#         inter_x_min = max(x1_min, x2_min)
#         inter_y_min = max(y1_min, y2_min)
#         inter_x_max = min(x1_max, x2_max)
#         inter_y_max = min(y1_max, y2_max)

#         # Compute the area of intersection
#         inter_area = max(0, inter_x_max - inter_x_min + 1) * max(0, inter_y_max - inter_y_min + 1)

#         # Compute the area of both bounding boxes
#         box1_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
#         box2_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)

#         # Compute the IoU
#         iou = inter_area / float(box1_area + box2_area - inter_area)
#         return iou

#     def smallest_enclosing_box(self, boxes: torch.Tensor) -> np.ndarray:
#         """Compute the Convex Hull for a set of predicted boxes and return the enclosing points."""
#         if boxes.size(0) == 0:
#             return np.array([])
#         points = []
#         for box in boxes:
#             x_min, y_min, x_max, y_max = box.tolist()
#             points.append([x_min, y_min])
#             points.append([x_max, y_min])
#             points.append([x_min, y_max])
#             points.append([x_max, y_max])
#         points = np.array(points)
#         hull = ConvexHull(points)
#         return points[hull.vertices]

#     def find_intersecting_boxes(self, gt_box: List[int], predictions: List[List[int]]) -> List[List[int]]:
#         """Find predicted boxes that intersect with a given ground truth box based on IoU."""
#         intersecting_boxes = []
#         for pred_box in predictions:
#             iou = self.calculate_iou(gt_box, pred_box)
#             if iou >= self.iou_threshold:
#                 intersecting_boxes.append(pred_box)
#         return intersecting_boxes

#     def process_files(self, ground_truth_file: str, prediction_file: str) -> Dict[Tuple[int, int, int, int], List[List[int]]]:
#         ground_truth = self.read_boxes_from_file(ground_truth_file)
#         predictions = self.read_boxes_from_file(prediction_file)
#         intersection_dict = {}
#         for gt_box in ground_truth:
#             intersecting_boxes = self.find_intersecting_boxes(gt_box, predictions)
#             intersection_dict[tuple(gt_box)] = intersecting_boxes
#         return intersection_dict

#     def calculate_recall(self, ground_truth_file: str, prediction_file: str, use_convex_hull: bool = False) -> float:
#         """Calculate Recall with or without Convex Hull preprocessing."""
#         intersection_dict = self.process_files(ground_truth_file, prediction_file)
#         total_gt_boxes = len(intersection_dict)
#         true_positives = 0

#         for gt_box, intersecting_preds in intersection_dict.items():
#             if use_convex_hull and len(intersecting_preds) > 1:
#                 # Apply Convex Hull to the intersecting predictions
#                 hull_points = self.smallest_enclosing_box(torch.tensor(intersecting_preds))
#                 if len(hull_points) > 0 and self.calculate_iou(gt_box, hull_points.tolist()) >= self.iou_threshold:
#                     true_positives += 1
#             else:
#                 # If any prediction intersects with the ground truth box, it's a true positive
#                 if len(intersecting_preds) > 0:
#                     true_positives += 1

#         recall = true_positives / total_gt_boxes if total_gt_boxes > 0 else 0
#         return recall

# # Example Usage
# image_width = 1800
# image_height = 4000
# ground_truth_file = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/test_data/gt/1192.txt'
# prediction_file = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/yolo_result/labels/target_1192.txt'

# evaluator = BoundingBoxEvaluator(image_width, image_height)

# # Recall without Convex Hull
# recall_without_hull = evaluator.calculate_recall(ground_truth_file, prediction_file, use_convex_hull=False)
# print(f"Recall without Convex Hull: {recall_without_hull}")

# # Recall with Convex Hull
# recall_with_hull = evaluator.calculate_recall(ground_truth_file, prediction_file, use_convex_hull=True)
# print(f"Recall with Convex Hull: {recall_with_hull}")


# #NOTE: WORKING
# import numpy as np
# import cv2
# from shapely.geometry import Polygon

# def denormalize_yolo_bbox(yolo_bbox, img_width, img_height):
#     """
#     Convert YOLO format (normalized x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
#     where the input values are relative to the image dimensions (normalized between 0 and 1).
#     """
#     x_center, y_center, width, height = yolo_bbox
#     x_min = int((x_center - width / 2) * img_width)
#     y_min = int((y_center - height / 2) * img_height)
#     x_max = int((x_center + width / 2) * img_width)
#     y_max = int((y_center + height / 2) * img_height)
#     return [x_min, y_min, x_max, y_max]

# def yolo_to_polygons(yolo_file, img_width, img_height):
#     """
#     Load ground truth bounding boxes from YOLO format file (normalized values) 
#     and convert them to rectangular polygons in pixel coordinates.
#     """
#     polygons = []
#     with open(yolo_file, 'r') as f:
#         for line in f:
#             bbox = list(map(float, line.strip().split()[1:]))  # Ignore class ID
#             x_min, y_min, x_max, y_max = denormalize_yolo_bbox(bbox, img_width, img_height)
#             # Create a polygon from the bounding box
#             polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
#             polygons.append(polygon)
#     return polygons

# def calculate_iou(ground_truth_polygon, prediction_polygon):
#     """
#     Calculate IoU between a ground truth rectangular polygon and a predicted polygon.
#     """
#     if not ground_truth_polygon.is_valid or not prediction_polygon.is_valid:
#         return 0.0

#     intersection_area = ground_truth_polygon.intersection(prediction_polygon).area
#     union_area = ground_truth_polygon.union(prediction_polygon).area
#     return intersection_area / union_area if union_area > 0 else 0.0

# def calculate_recall(ground_truth_file, prediction_polygons, img_width, img_height, iou_threshold=0.95):
#     """
#     Calculate recall based on IoU between ground truth (YOLO format) and predicted convex hull polygons.
#     The ground truth is provided in normalized format.
#     """
#     # Load ground truth polygons from YOLO format file
#     ground_truth_polygons = yolo_to_polygons(ground_truth_file, img_width, img_height)
    
#     # Count the number of true positives (correctly matched ground truth boxes)
#     true_positives = 0
    
#     for gt_polygon in ground_truth_polygons:
#         # Check if any prediction matches the ground truth box with IoU >= threshold
#         matched = any(calculate_iou(gt_polygon, pred_polygon) >= iou_threshold for pred_polygon in prediction_polygons)
#         if matched:
#             true_positives += 1

#     recall = true_positives / len(ground_truth_polygons) if ground_truth_polygons else 0.0
#     return recall

# def load_prediction_polygons(prediction_file):
#     """
#     Load predicted convex hull or polygons from a file. Assume each line contains a list of points.
#     Example format:
#     100,150 120,200 160,180
#     """
#     polygons = []
#     with open(prediction_file, 'r') as f:
#         for line in f:
#             points= np.array([[int(x) for x in coord] for coord in eval(line[1:-2])]) #np.array(line[1:-2])
#             print(points, "\n")
#             # points = np.array([list(map(int, p[1:-2].split(','))) for p in line.strip()])
#             # print(points)
#             polygons.append(Polygon(points))
#     return polygons

# # Example usage:
# ground_truth_file = "/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/test_data/gt/1192.txt"
# prediction_file = "/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/aa.txt"
# img_width, img_height = 1800, 4000

# # Load predictions (list of Polygon objects)
# prediction_polygons = load_prediction_polygons(prediction_file)

# # Calculate recall based on IoU
# recall = calculate_recall(ground_truth_file, prediction_polygons, img_width, img_height, iou_threshold=0.05)
# print(f"Recall: {recall:.4f}")


# import os
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from shapely.geometry import Polygon

# def denormalize_yolo_bbox(yolo_bbox, img_width, img_height):
#     x_center, y_center, width, height = yolo_bbox
#     x_min = int((x_center - width / 2) * img_width)
#     y_min = int((y_center - height / 2) * img_height)
#     x_max = int((x_center + width / 2) * img_width)
#     y_max = int((y_center + height / 2) * img_height)
#     return [x_min, y_min, x_max, y_max]

# def yolo_to_polygons(yolo_file, img_width, img_height):
#     polygons = []
#     with open(yolo_file, 'r') as f:
#         for line in f:
#             bbox = list(map(float, line.strip().split()[1:]))  # Ignore class ID
#             x_min, y_min, x_max, y_max = denormalize_yolo_bbox(bbox, img_width, img_height)
#             polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
#             polygons.append(polygon)
#     return polygons

# def calculate_iou(ground_truth_polygon, prediction_polygon):
#     if not ground_truth_polygon.is_valid or not prediction_polygon.is_valid:
#         return 0.0
#     intersection_area = ground_truth_polygon.intersection(prediction_polygon).area
#     union_area = ground_truth_polygon.union(prediction_polygon).area
#     return intersection_area / union_area if union_area > 0 else 0.0

# def calculate_overall_recall(ground_truth_file, prediction_polygons, img_width, img_height, iou_threshold):
#     ground_truth_polygons = yolo_to_polygons(ground_truth_file, img_width, img_height)
    
#     true_positives = 0
#     for gt_polygon in ground_truth_polygons:
#         matched = any(calculate_iou(gt_polygon, pred_polygon) >= iou_threshold for pred_polygon in prediction_polygons)
#         if matched:
#             true_positives += 1

#     return true_positives, len(ground_truth_polygons)

# def load_prediction_polygons(prediction_file):
#     polygons = []
#     with open(prediction_file, 'r') as f:
#         for line in f:
#             points = np.array([[int(x) for x in coord] for coord in eval(line[1:-2])])
#             polygons.append(Polygon(points))
#     return polygons

# def process_directory_for_recall_vs_iou(gt_dir, hull_points_dir, img_dir, iou_thresholds):
#     total_true_positives_per_threshold = np.zeros(len(iou_thresholds))
#     total_ground_truths = 0

#     for filename in os.listdir(gt_dir):
#         if filename.endswith(".txt"):  # Assuming ground truth files are text files
#             base_filename = filename.split('.')[0]

#             gt_file = os.path.join(gt_dir, filename)
#             prediction_file = os.path.join(hull_points_dir, f"hull_points_{base_filename}.txt")
#             image_file = os.path.join(img_dir, f"{base_filename}.jpg")

#             if os.path.exists(prediction_file) and os.path.exists(image_file):
#                 # Get image dimensions
#                 img = cv2.imread(image_file)
#                 img_height, img_width, _ = img.shape

#                 # Load prediction polygons
#                 prediction_polygons = load_prediction_polygons(prediction_file)

#                 # Calculate true positives and total ground truth boxes for each IoU threshold
#                 for i, iou_threshold in enumerate(iou_thresholds):
#                     true_positives, ground_truth_count = calculate_overall_recall(gt_file, prediction_polygons, img_width, img_height, iou_threshold)
#                     total_true_positives_per_threshold[i] += true_positives

#                 total_ground_truths += ground_truth_count

#     # Calculate recall for each IoU threshold
#     recalls = total_true_positives_per_threshold / total_ground_truths if total_ground_truths > 0 else np.zeros(len(iou_thresholds))
#     return recalls

# def plot_recall_vs_iou(iou_thresholds, recalls):
#     plt.figure(figsize=(8, 6))
#     plt.plot(iou_thresholds, recalls, marker='o', linestyle='-', color='b')
#     plt.title('Recall vs IoU Threshold')
#     plt.xlabel('IoU Threshold')
#     plt.ylabel('Recall')
#     plt.grid(True)
#     plt.savefig("./recall_vs_iou")

# if __name__ == "__main__":
#     gt_dir = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/test_data/gt'
#     hull_points_dir = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/output'
#     img_dir = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/test_data/images'

#     # Define a range of IoU thresholds
#     iou_thresholds = np.linspace(0.1, 0.95, 10)

#     # Process the entire directory and calculate recalls for each IoU threshold
#     recalls = process_directory_for_recall_vs_iou(gt_dir, hull_points_dir, img_dir, iou_thresholds)

#     # Plot recall vs IoU threshold
#     plot_recall_vs_iou(iou_thresholds, recalls)



import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

def denormalize_yolo_bbox(yolo_bbox, img_width, img_height):
    x_center, y_center, width, height = yolo_bbox
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    x_max = int((x_center + width / 2) * img_width)
    y_max = int((y_center + height / 2) * img_height)
    return [x_min, y_min, x_max, y_max]

def yolo_to_polygons(yolo_file, img_width, img_height):
    """
    Load YOLO predictions (ignoring the class label) and convert them to rectangular polygons.
    """
    polygons = []
    with open(yolo_file, 'r') as f:
        for line in f:
            bbox = list(map(float, line.strip().split()[1:]))  # Ignore class ID
            x_min, y_min, x_max, y_max = denormalize_yolo_bbox(bbox, img_width, img_height)
            polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
            polygons.append(polygon)
    return polygons

def load_prediction_polygons(prediction_file):
    """
    Load predicted convex hull polygons from a file.
    """
    polygons = []
    with open(prediction_file, 'r') as f:
        for line in f:
            points = np.array([[int(x) for x in coord] for coord in eval(line[1:-2])])
            polygons.append(Polygon(points))
    return polygons

def calculate_iou(ground_truth_polygon, prediction_polygon):
    """
    Calculate IoU between ground truth and predicted polygons.
    """
    if not ground_truth_polygon.is_valid or not prediction_polygon.is_valid:
        return 0.0
    intersection_area = ground_truth_polygon.intersection(prediction_polygon).area
    union_area = ground_truth_polygon.union(prediction_polygon).area
    return intersection_area / union_area if union_area > 0 else 0.0

def calculate_recall_for_threshold(ground_truth_file, prediction_polygons, img_width, img_height, iou_threshold):
    """
    Calculate recall for a single image at a specific IoU threshold.
    """
    ground_truth_polygons = yolo_to_polygons(ground_truth_file, img_width, img_height)
    
    true_positives = 0
    total_ground_truth = len(ground_truth_polygons)
    
    for gt_polygon in ground_truth_polygons:
        matched = any(calculate_iou(gt_polygon, pred_polygon) >= iou_threshold for pred_polygon in prediction_polygons)
        if matched:
            true_positives += 1

    return true_positives, total_ground_truth

def process_directory_for_recall_comparison(gt_dir, hull_points_dir, raw_predictions_dir, img_dir, iou_thresholds):
    total_true_positives_hull = np.zeros(len(iou_thresholds))
    total_true_positives_raw = np.zeros(len(iou_thresholds))
    total_ground_truths = 0

    for filename in os.listdir(gt_dir):
        if filename.endswith(".txt"):
            base_filename = filename.split('.')[0]

            gt_file = os.path.join(gt_dir, filename)
            hull_prediction_file = os.path.join(hull_points_dir, f"hull_points_{base_filename}.txt")
            raw_prediction_file = os.path.join(raw_predictions_dir, f"{base_filename}.txt")
            image_file = os.path.join(img_dir, f"{base_filename}.jpg")

            if os.path.exists(hull_prediction_file) and os.path.exists(raw_prediction_file) and os.path.exists(image_file):
                # Get image dimensions
                img = cv2.imread(image_file)
                img_height, img_width, _ = img.shape

                # Load prediction polygons for hull points and raw predictions
                hull_prediction_polygons = load_prediction_polygons(hull_prediction_file)
                raw_prediction_polygons = yolo_to_polygons(raw_prediction_file, img_width, img_height)

                # Debugging: Check if predictions are being loaded correctly
                print(f"Processing {base_filename}:")
                print(f"Raw Predictions: {len(raw_prediction_polygons)} polygons")
                print(f"Hull Points: {len(hull_prediction_polygons)} polygons")

                # Calculate recall for both hull points and raw predictions for each IoU threshold
                for i, iou_threshold in enumerate(iou_thresholds):
                    true_positives_hull, ground_truth_count = calculate_recall_for_threshold(gt_file, hull_prediction_polygons, img_width, img_height, iou_threshold)
                    true_positives_raw, _ = calculate_recall_for_threshold(gt_file, raw_prediction_polygons, img_width, img_height, iou_threshold)

                    total_true_positives_hull[i] += true_positives_hull
                    total_true_positives_raw[i] += true_positives_raw

                total_ground_truths += ground_truth_count

    # Calculate recall for hull points and raw predictions for each IoU threshold
    recall_hull = total_true_positives_hull / total_ground_truths if total_ground_truths > 0 else np.zeros(len(iou_thresholds))
    recall_raw = total_true_positives_raw / total_ground_truths if total_ground_truths > 0 else np.zeros(len(iou_thresholds))

    return recall_hull, recall_raw

def plot_comparative_recall(iou_thresholds, recall_hull, recall_raw):
    plt.figure(figsize=(8, 6))
    plt.plot(iou_thresholds, recall_hull, marker='o', linestyle='-', color='b', label='Hull Points')
    plt.plot(iou_thresholds, recall_raw, marker='x', linestyle='--', color='r', label='Raw Predictions')
    plt.title('Comparative Recall: Hull Points vs Raw Predictions')
    plt.xlabel('IoU Threshold')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    plt.savefig("./hull_vs_woHull.png")

if __name__ == "__main__":
    # gt_dir = '/path/to/ground_truth_directory'
    # hull_points_dir = '/path/to/hull_points_directory'
    # raw_predictions_dir = '/path/to/raw_predictions_directory'
    # img_dir = '/path/to/image_directory'

    gt_dir = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/test_data/gt'
    hull_points_dir = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/output'
    img_dir = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/test_data/images'
    raw_predictions_dir = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/test_data/labels/labels'
    # Define a range of IoU thresholds
    iou_thresholds = np.linspace(0.1, 0.95, 10)

    # Process the directory and calculate recalls for hull points and raw predictions
    recall_hull, recall_raw = process_directory_for_recall_comparison(gt_dir, hull_points_dir, raw_predictions_dir, img_dir, iou_thresholds)

    # Plot comparative recall
    plot_comparative_recall(iou_thresholds, recall_hull, recall_raw)
