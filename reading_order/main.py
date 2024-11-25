#@Authoer: Akash
#@Date: 2024-11-25
#@Last Modified by: Akash

import os
import sys
import cv2
import json
from ultralytics import YOLO
from yolo_pred import YOLOJsonExtractor
# from reading_order import get_reading_order, draw_bounind_boxes

from reading_order import BoundingBoxVisualizer




if __name__ == "__main__":
    input_dir = "/home/akash/ws/dataset/hand_written/finetune_data/telugu_test/images/val/"
    output_dir = "./result"

    os.makedirs(output_dir, exist_ok=True)


    # Initialize the class with the model path
    model_path = "/home/akash/ws/artifacts/HW/HW_telugu_v02_081024/HW_telugu_v02_081024_2/weights/best.pt"
    # image_path = "/home/akash/ws/dataset/hand_written/finetune_data/telugu_test/images/val/24300.jpg"
    # page_number = 1

    # Create an instance of the class and process the image
    yolo_extractor = YOLOJsonExtractor(model_path)
    visualizer = BoundingBoxVisualizer(delta_y= 50)

    # Iterate through all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Process the image with YOLO extractor
            page_number = 1  # Page number for the bounding boxes (static in this example)
            result_json = yolo_extractor.process_image(image_path, page_number)

            # Assign reading order
            ordered_boxes = visualizer.get_reading_order(result_json)

            # Draw bounding boxes and save the output
            visualizer.draw_bounding_boxes(image_path=image_path, bounding_boxes=ordered_boxes, output_path=output_path)

    # result_json = yolo_extractor.process_image(image_path, page_number)

    # # Print the resulting JSON data
    # print(json.dumps(result_json, indent=4, ensure_ascii=False))

    

    # #Ordered box
    # ordered_boxes = visualizer.get_reading_order(result_json)

    # # Draw bounding boxes on the image
    # visualizer.draw_bounding_boxes(
    #     image_path=image_path,
    #     bounding_boxes=ordered_boxes,
    #     output_path="./after.jpeg"
    # )