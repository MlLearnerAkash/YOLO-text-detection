# import json
# import matplotlib.pyplot as plt
# import cv2

# import matplotlib.patches as patches


# def draw_bounding_boxes(image_path, bounding_boxes, output_path):
#     """
#     Draw bounding boxes and optional reading order on a given image using OpenCV and save the output.

#     Args:
#         image_path (str): Path to the input image.
#         bounding_boxes (list): List of bounding boxes with `x_min`, `y_min`, `x_max`, `y_max`, and optional `reading_order`.
#         output_path (str): Path to save the image with drawn bounding boxes.
#     """
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Image not found at {image_path}")
    
#     # Font and scale settings for text
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1.5
#     font_thickness = 3
#     text_color = (0, 255, 0)  # Green color for text in BGR
    
#     # Draw each bounding box
#     for box_info in bounding_boxes:
#         bbox = box_info["bounding_box"]
#         # Draw the rectangle
#         cv2.rectangle(
#             image,
#             (int(bbox["x_min"]), int(bbox["y_min"])),
#             (int(bbox["x_max"]), int(bbox["y_max"])),
#             color=(255, 0, 0),  # Blue color in BGR format
#             thickness=2
#         )
        
#         # Draw the reading order if present
#         if "reading_order" in box_info:
#             reading_order = str(box_info["reading_order"])
#             print()
#             cv2.putText(
#                 image,
#                 reading_order,
#                 (int(bbox["x_min"]), int(bbox["y_min"]) - 15),  # Position above the top-left corner
#                 font,
#                 font_scale,
#                 text_color,
#                 thickness=font_thickness,
#                 lineType=cv2.LINE_AA
#             )
    
#     # Save the resulting image
#     cv2.imwrite(output_path, image)
#     print(f"Image with bounding boxes and reading order saved to {output_path}")


# def get_reading_order(bounding_boxes, delta_y=50):
#     # Step 1: Sort bounding boxes by `y_min` to group by rows
#     bounding_boxes = sorted(bounding_boxes["words"], key=lambda box: box["bounding_box"]['y_min'])
#     rows = []
#     current_row = [bounding_boxes[0]]
    
#     # Step 2: Group boxes into rows based on `delta_y` threshold
#     for box in bounding_boxes[1:]:
#         if abs(box["bounding_box"]['y_min'] - current_row[-1]["bounding_box"]['y_min']) < delta_y:
#             current_row.append(box)
#         else:
#             rows.append(current_row)
#             current_row = [box]
#     rows.append(current_row)  # Add the last row
    
#     # Step 3: Sort each row by `x_min` (left to right)
#     ordered_boxes = []
#     reading_order = 1
    
#     for row in rows:
#         row_sorted = sorted(row, key=lambda box: box["bounding_box"]['x_min'])
#         for box in row_sorted:
#             box['reading_order'] = reading_order
#             ordered_boxes.append(box)
#             reading_order += 1
    
#     return ordered_boxes


# if __name__ == "__main__":
#     # Opening the JSON file
#     with open('/home/akash/ws/YOLO-text-detection/ultralytics/output.json', 'r') as f:
#         bounding_boxes = json.load(f)
    
#  # Get the reading order of bounding boxes
#     ordered_boxes = get_reading_order(bounding_boxes)
 
#     draw_bounding_boxes(image_path= "/home/akash/ws/dataset/hand_written/finetune_data/telugu_test/images/val/24300.jpg",
#                         bounding_boxes= ordered_boxes, output_path= "./after.jpeg")

import json
import cv2


class BoundingBoxVisualizer:
    """
    A class to visualize bounding boxes, assign reading order, and draw them on an image.
    """

    def __init__(self, delta_y=50):
        """
        Initialize the BoundingBoxVisualizer with parameters for grouping rows.

        Args:
            delta_y (int): Threshold for grouping bounding boxes into rows based on vertical proximity.
        """
        self.delta_y = delta_y

    def get_reading_order(self, bounding_boxes):
        """
        Assign reading order to bounding boxes.

        Args:
            bounding_boxes (dict): JSON data with bounding boxes.

        Returns:
            list: List of bounding boxes with `reading_order` assigned.
        """
        # Step 1: Sort bounding boxes by `y_min` to group by rows
        bounding_boxes = sorted(bounding_boxes["words"], key=lambda box: box["bounding_box"]['y_min'])
        rows = []
        current_row = [bounding_boxes[0]]

        # Step 2: Group boxes into rows based on `delta_y` threshold
        for box in bounding_boxes[1:]:
            if abs(box["bounding_box"]['y_min'] - current_row[-1]["bounding_box"]['y_min']) < self.delta_y:
                current_row.append(box)
            else:
                rows.append(current_row)
                current_row = [box]
        rows.append(current_row)  # Add the last row

        # Step 3: Sort each row by `x_min` (left to right)
        ordered_boxes = []
        reading_order = 1

        for row in rows:
            row_sorted = sorted(row, key=lambda box: box["bounding_box"]['x_min'])
            for box in row_sorted:
                box['reading_order'] = reading_order
                ordered_boxes.append(box)
                reading_order += 1

        return ordered_boxes

    def draw_bounding_boxes(self, image_path, bounding_boxes, output_path):
        """
        Draw bounding boxes and optional reading order on a given image using OpenCV and save the output.

        Args:
            image_path (str): Path to the input image.
            bounding_boxes (list): List of bounding boxes with `x_min`, `y_min`, `x_max`, `y_max`, and optional `reading_order`.
            output_path (str): Path to save the image with drawn bounding boxes.
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Font and scale settings for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 3
        text_color = (0, 255, 0)  # Green color for text in BGR
        
        # Draw each bounding box
        for box_info in bounding_boxes:
            bbox = box_info["bounding_box"]
            # Draw the rectangle
            cv2.rectangle(
                image,
                (int(bbox["x_min"]), int(bbox["y_min"])),
                (int(bbox["x_max"]), int(bbox["y_max"])),
                color=(255, 0, 0),  # Blue color in BGR format
                thickness=2
            )
            
            # Draw the reading order if present
            if "reading_order" in box_info:
                reading_order = str(box_info["reading_order"])
                cv2.putText(
                    image,
                    reading_order,
                    (int(bbox["x_min"]), int(bbox["y_min"]) - 15),  # Position above the top-left corner
                    font,
                    font_scale,
                    text_color,
                    thickness=font_thickness,
                    lineType=cv2.LINE_AA
                )
        
        # Save the resulting image
        cv2.imwrite(output_path, image)
        print(f"Image with bounding boxes and reading order saved to {output_path}")


if __name__ == "__main__":
    # Example usage of the BoundingBoxVisualizer class
    visualizer = BoundingBoxVisualizer(delta_y=50)

    # Open the JSON file containing bounding boxes
    with open('/home/akash/ws/YOLO-text-detection/ultralytics/output.json', 'r') as f:
        bounding_boxes = json.load(f)

    # Get the reading order of bounding boxes
    ordered_boxes = visualizer.get_reading_order(bounding_boxes)

    # Draw bounding boxes on the image
    visualizer.draw_bounding_boxes(
        image_path="/home/akash/ws/dataset/hand_written/finetune_data/telugu_test/images/val/24300.jpg",
        bounding_boxes=ordered_boxes,
        output_path="./after.jpeg"
    )
