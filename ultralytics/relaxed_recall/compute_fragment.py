from typing import List, Dict, Tuple
import torch
import math
import json
import cv2
from scipy.spatial import ConvexHull
import numpy as np
# Function to read bounding boxes from a text file
def read_boxes_from_file(file_path: str) -> List[List[int]]:
    boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            box = list(map(float, line.strip().split(' ')))[1:]  # YOLO format: ignoring class id
            box = denormalize_yolo_bbox(box,1800, 4000)
            boxes.append(box)
    return boxes

# Function to denormalize YOLO bounding box format to standard bounding box format
def denormalize_yolo_bbox(yolo_bbox: List[float], image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    """
    Convert YOLO format (x_center, y_center, width, height) to normal (x_min, y_min, x_max, y_max)
    
    Parameters:
        yolo_bbox (List[float]): A list containing [x_center, y_center, width, height] in YOLO format (normalized values).
        image_width (int): The width of the image.
        image_height (int): The height of the image.
        
    Returns:
        Tuple[int, int, int, int]: Denormalized (x_min, y_min, x_max, y_max) in image pixel coordinates.
    """
    x_center, y_center, width, height = yolo_bbox
    
    # Convert center x, y and width, height from normalized to pixel values
    x_center_pixel = x_center * image_width
    y_center_pixel = y_center * image_height
    width_pixel = width * image_width
    height_pixel = height * image_height
    
    # Calculate top-left (x_min, y_min) and bottom-right (x_max, y_max) coordinates
    x_min = int(x_center_pixel - (width_pixel / 2))
    y_min = int(y_center_pixel - (height_pixel / 2))
    x_max = int(x_center_pixel + (width_pixel / 2))
    y_max = int(y_center_pixel + (height_pixel / 2))
    
    return x_min, y_min, x_max, y_max
#TODO: Change to centre based iou
# # Function to calculate Intersection over Union (IoU)
# def compute_iou(box1: List[int], box2: List[int]) -> float:
#     x1, y1, x2, y2 = min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3])
    
#     # Compute the area of the intersection
#     intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
#     # Compute the area of both the prediction and ground truth rectangles
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
#     # Compute the union area
#     union_area = box1_area + box2_area - intersection_area
    
#     # Compute the IoU
#     iou = intersection_area / box2_area if union_area != 0 else 0
#     return iou

# Function to check if the center of box2 is inside box1
def is_center_inside(box1: List[int], box2: List[int]) -> bool:
    # Calculate the center of box2
    box2_center_x = (box2[0] + box2[2]) / 2
    box2_center_y = (box2[1] + box2[3]) / 2
    
    # Check if the center of box2 is inside box1
    if (box1[0] <= box2_center_x <= box1[2]) and (box1[1] <= box2_center_y <= box1[3]):
        return True
    else:
        return False

def compute_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    box1 = torch.tensor(box1)
    box2 = torch.tensor(box2)
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    box2_area = w2 * h2
    iou = inter / box2_area

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def smallest_enclosing_box(boxes: torch.Tensor) -> np.ndarray:
    """
    Compute the convex hull that encloses all bounding boxes.
    
    Parameters:
        boxes (torch.Tensor): A tensor of shape (N, 4), where N is the number of bounding boxes.
                              Each box is in the format [x_min, y_min, x_max, y_max].
    
    Returns:
        np.ndarray: An array of points representing the convex hull.
    """
    # Ensure there is at least one box
    if boxes.size(0) == 0:
        raise ValueError("No bounding boxes provided.")
    
    # Collect the corner points of all bounding boxes
    points = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box.tolist()
        
        # Add the four corners of each bounding box
        points.append([x_min, y_min])  # Bottom-left corner
        points.append([x_max, y_min])  # Bottom-right corner
        points.append([x_min, y_max])  # Top-left corner
        points.append([x_max, y_max])  # Top-right corner

    # Convert points to a numpy array
    points = np.array(points)
    # Compute the convex hull
    hull = ConvexHull(points)

    # Return the coordinates of the convex hull vertices
    return points[hull.vertices]


# def annotate_image_with_bbox(image_path: str, bbox: torch.Tensor, output_image_path: str):
#     """
#     Annotate the image with the bounding box.

#     Parameters:
#         image_path (str): Path to the input image.
#         bbox (torch.Tensor): Bounding box to annotate in format [x_min, y_min, x_max, y_max].
#         output_image_path (str): Path to save the annotated image.
#     """
#     # Read the image using OpenCV
#     image = cv2.imread(image_path)

#     # Convert points to the required format for OpenCV (integer coordinates)
#     convex_hull = bbox.astype(int)

#     # Draw the convex hull on the image as a polygon
#     cv2.polylines(image, [convex_hull], isClosed=True, color=(0, 255, 0), thickness=2)

#     # Optionally, fill the polygon with a transparent color
#     # cv2.fillPoly(image, [convex_hull], color=(0, 255, 0, 50))

#     # Save the output image
#     cv2.imwrite(output_image_path, image)



def annotate_image_with_bbox(image_path: str, bboxes: List[torch.Tensor], output_image_path: str):
    """
    Annotate the image with multiple bounding boxes.

    Parameters:
        image_path (str): Path to the input image.
        bboxes (list of torch.Tensor): List of bounding boxes to annotate, each in format [x_min, y_min, x_max, y_max].
        output_image_path (str): Path to save the annotated image.
    """
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Iterate through each bounding box
    for bbox in bboxes:
        # Convert points to the required format for OpenCV (integer coordinates)
        convex_hull = bbox.astype(int)

        # Draw the convex hull on the image as a polygon
        cv2.polylines(image, [convex_hull], isClosed=True, color=(0, 255, 0), thickness=2)

        # Optionally, fill the polygon with a transparent color
        # cv2.fillPoly(image, [convex_hull], color=(0, 255, 0, 50))

    # Save the output image
    cv2.imwrite(output_image_path, image)
    print(f"Annotated image saved at {output_image_path}")
  
# Function to find predictions that completely intersect with a specific ground truth box
def find_complete_intersections_with_gt_box(gt_box: List[int], predictions: List[List[int]]) -> List[List[int]]:
    complete_intersections = []
    
    # Iterate over all predicted bounding boxes
    for pred_box in predictions:
        iou = is_center_inside(gt_box, pred_box)
        
        # If IoU is 1, we consider it as a complete intersection
        if iou :
            complete_intersections.append(pred_box)
    
    return complete_intersections

# Main function to read files and return complete intersections for each ground truth box
def process_files(ground_truth_file: str, prediction_file: str) -> Dict[Tuple[int, int, int, int], List[List[int]]]:
    # Read ground truth and predictions from files
    ground_truth = read_boxes_from_file(ground_truth_file)
    predictions = read_boxes_from_file(prediction_file)
    
    complete_intersections_dict = {}
    
    # Iterate through each ground truth box
    for gt_box in ground_truth:
        # Find predictions with complete intersections for the current ground truth box
        complete_intersections = find_complete_intersections_with_gt_box(gt_box, predictions)
        
        # Store the ground truth box as the key and the list of complete intersections as the value
        complete_intersections_dict[tuple(gt_box)] = complete_intersections
        # if len(complete_intersections)> 0:
    
    return complete_intersections_dict

# Example Usage
ground_truth_file = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/test_data/gt/1192.txt'  # Path to the ground truth file
prediction_file = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/yolo_result/labels/target_1192.txt'  # Path to the predictions file

result = process_files(ground_truth_file, prediction_file)

hull_points = []
for val in list(result.values()):
    hull_point = smallest_enclosing_box(torch.tensor(val))
    hull_points.append(hull_point)

annotate_image_with_bbox("/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/test_data/target_1192.jpg", 
                        # smallest_enclosing_box(torch.tensor(list(result.values()))), 
                        hull_points,
                        "./target_1192_anno.jpg")
print("Complete intersections between predictions and ground truth boxes:")
# for gt_box, intersecting_preds in result.items():
#     print(f"Ground Truth Box {gt_box}: {intersecting_preds}")

with open("mapping.json", "w") as f:
    json.dump(str(result), f)
