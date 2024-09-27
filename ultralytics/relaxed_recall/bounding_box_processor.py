from typing import List, Dict, Tuple
import torch
import math
import json
import cv2
from scipy.spatial import ConvexHull
import numpy as np
import os

class BoundingBoxProcessor:
    def __init__(self, image_width: int = 4000, image_height: int = 1800):
        self.image_width = image_width
        self.image_height = image_height

    # Function to read bounding boxes from a text file
    def read_boxes_from_file(self, file_path: str) -> List[List[int]]:
        boxes = []
        with open(file_path, 'r') as file:
            for line in file:
                box = list(map(float, line.strip().split(' ')))[1:]  # YOLO format: ignoring class id
                box = self.denormalize_yolo_bbox(box)
                boxes.append(box)
        return boxes

    # Function to denormalize YOLO bounding box format to standard bounding box format
    def denormalize_yolo_bbox(self, yolo_bbox: List[float]) -> Tuple[int, int, int, int]:
        x_center, y_center, width, height = yolo_bbox

        # Convert center x, y and width, height from normalized to pixel values
        x_center_pixel = x_center * self.image_width
        y_center_pixel = y_center * self.image_height
        width_pixel = width * self.image_width
        height_pixel = height * self.image_height

        # Calculate top-left (x_min, y_min) and bottom-right (x_max, y_max) coordinates
        x_min = int(x_center_pixel - (width_pixel / 2))
        y_min = int(y_center_pixel - (height_pixel / 2))
        x_max = int(x_center_pixel + (width_pixel / 2))
        y_max = int(y_center_pixel + (height_pixel / 2))

        return x_min, y_min, x_max, y_max

    # Function to check if the center of box2 is inside box1
    def is_center_inside(self, box1: List[int], box2: List[int]) -> bool:
        box2_center_x = (box2[0] + box2[2]) / 2
        box2_center_y = (box2[1] + box2[3]) / 2

        return box1[0] <= box2_center_x <= box1[2] and box1[1] <= box2_center_y <= box1[3]

    # Function to compute the convex hull for bounding boxes
    def smallest_enclosing_box(self, boxes: torch.Tensor) -> np.ndarray:
        if boxes.size(0) == 0:
            raise ValueError("No bounding boxes provided.")

        points = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box.tolist()
            points.append([x_min, y_min])
            points.append([x_max, y_min])
            points.append([x_min, y_max])
            points.append([x_max, y_max])

        points = np.array(points)
        hull = ConvexHull(points)
        return points[hull.vertices]

    # Function to annotate image with bounding boxes
    def annotate_image_with_bbox(self, image_path: str, bboxes: List[torch.Tensor], output_image_path: str):
        image = cv2.imread(image_path)
        for bbox in bboxes:
            convex_hull = bbox.astype(int)
            cv2.polylines(image, [convex_hull], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imwrite(output_image_path, image)
        print(f"Annotated image saved at {output_image_path}")

    # Function to find predictions that completely intersect with a specific ground truth box
    def find_complete_intersections_with_gt_box(self, gt_box: List[int], predictions: List[List[int]]) -> List[List[int]]:
        complete_intersections = []
        for pred_box in predictions:
            if self.is_center_inside(gt_box, pred_box):
                complete_intersections.append(pred_box)
        return complete_intersections

    # Main function to read files and return complete intersections for each ground truth box
    def process_files(self, ground_truth_file: str, prediction_file: str) -> Dict[Tuple[int, int, int, int], List[List[int]]]:
        ground_truth = self.read_boxes_from_file(ground_truth_file)
        predictions = self.read_boxes_from_file(prediction_file)

        complete_intersections_dict = {}
        for gt_box in ground_truth:
            complete_intersections = self.find_complete_intersections_with_gt_box(gt_box, predictions)
            complete_intersections_dict[tuple(gt_box)] = complete_intersections
        return complete_intersections_dict

    # Method to get convex hull points for an image
    def get_hull_points(self, ground_truth_file: str, prediction_file: str, image_path: str, output_image_path: str) -> List[np.ndarray]:
        result = self.process_files(ground_truth_file, prediction_file)
        hull_points = []

        for val in result.values():
            try:
                hull_point = self.smallest_enclosing_box(torch.tensor(val))
                hull_points.append(hull_point)
            except:
                pass
            

        # Annotate the image with the bounding boxes
        self.annotate_image_with_bbox(image_path, hull_points, output_image_path)
        return hull_points

    def write_polygon_points_to_file(self, hull_points: List[np.ndarray], output_file: str):
        """
        Writes the list of polygonal points (Convex Hull points) to a text file in a list-wise format.
        
        Parameters:
        hull_points (List[np.ndarray]): List of Convex Hull points, where each element is an array of coordinates.
        output_file (str): Path to the output text file.
        """
        with open(output_file, 'w') as file:
            for polygon in hull_points:
                # Convert each set of points (np.ndarray) to a list of integers
                points_list = polygon.tolist()
                # Write the list of points as a string, with each polygon on a new line
                file.write(f"{points_list}\n")
        print(f"Polygon points have been written to {output_file}")   



def process_directory(ground_truth_dir, prediction_dir, image_dir, output_dir, processor):
    for filename in os.listdir(prediction_dir):
        if filename.endswith(".txt"):  # Assuming the detection files are text files
            base_filename = filename.split('.')[0]

            # File paths
            ground_truth_file = os.path.join(ground_truth_dir, f"{base_filename}.txt")
            prediction_file = os.path.join(prediction_dir, f"{base_filename}.txt")
            image_path = os.path.join(image_dir, f"{base_filename}.jpg")
            image_shape = cv2.imread(image_path).shape
            #NOTE: intializing nside for loop
            processor = BoundingBoxProcessor(image_width=image_shape[1], image_height=image_shape[0])
            output_image_path = os.path.join(output_dir, f"annotated_{base_filename}.jpg")
            hull_points_file = os.path.join(output_dir, f"hull_points_{base_filename}.txt")

            # Ensure ground truth file exists before proceeding
            if os.path.exists(ground_truth_file) and os.path.exists(image_path):
                hull_points = processor.get_hull_points(ground_truth_file, prediction_file, image_path, output_image_path)
                processor.write_polygon_points_to_file(hull_points, hull_points_file)
                print(f"Hull points written for {base_filename}.")
# Example Usage
if __name__ == "__main__":
    # processor = BoundingBoxProcessor(image_width=1800, image_height=4000)

    # ground_truth_file = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/test_data/gt/1192.txt'
    # prediction_file = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/yolo_result/labels/target_1192.txt'
    # image_path = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/yolo_result/target_1192.jpg'
    # output_image_path = 'annotated_image.jpg'
    # hull_points_file = "hull_points.txt"

    # hull_points = processor.get_hull_points(ground_truth_file, prediction_file, image_path, output_image_path)
    # processor.write_polygon_points_to_file(hull_points, hull_points_file)
    # print("Hull Points per Image:", type(hull_points))

    if __name__ == "__main__":
        processor = BoundingBoxProcessor(image_width=1800, image_height=4000)

        ground_truth_dir = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/test_data/gt'
        prediction_dir = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/test_data/labels'
        image_dir = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/test_data/images'
        output_dir = '/home/akash/ws/YOLO-text-detection/ultralytics/relaxed_recall/output'

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        process_directory(ground_truth_dir, prediction_dir, image_dir, output_dir, processor)
