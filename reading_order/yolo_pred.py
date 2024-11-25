from ultralytics import YOLO
import json


class YOLOJsonExtractor:
    """
    A class to handle YOLO object detection and return bounding boxes in JSON format.
    """
    def __init__(self, model_path):
        """
        Initialize the YOLOJsonExtractor with a specified YOLO model.

        Args:
            model_path (str): Path to the YOLO model weights.
        """
        self.model = YOLO(model_path)

    def get_boxes_as_json(self, result, page_number):
        """
        Convert YOLO bounding boxes to JSON format without text or reading order.

        Args:
            result: YOLO model result object containing boxes.
            page_number (int): The page number corresponding to the detection.

        Returns:
            dict: JSON data with bounding boxes.
        """
        boxes = result.boxes  # Assuming this is a YOLOv8 result with a Boxes object
        xyxy_boxes = boxes.xyxy.cpu().numpy()  # Convert to numpy array (detach if on GPU)

        words = []
        for box in xyxy_boxes:
            x_min, y_min, x_max, y_max = box
            word_entry = {
                "bounding_box": {
                    "x_min": int(x_min),
                    "y_min": int(y_min),
                    "x_max": int(x_max),
                    "y_max": int(y_max)
                }
            }
            words.append(word_entry)

        output_data = {
            "page": page_number,
            "words": words
        }
        return output_data

    def process_image(self, image_path, page_number, conf=0.15, iou=0.15, img_size=(1024, 1024)):
        """
        Process an input image, run inference using YOLO, and return the results as JSON.

        Args:
            image_path (str): Path to the input image.
            page_number (int): Page number for the output JSON.
            conf (float): Confidence threshold for YOLO detection.
            iou (float): IoU threshold for YOLO detection.
            img_size (tuple): Image size to resize input images.

        Returns:
            dict: JSON data with bounding boxes for the image.
        """
        results = self.model(
            source=image_path,
            conf=conf,
            iou=iou,
            save_txt=False,
            save=False,
            imgsz=img_size,
            verbose=False
        )

        # Use the first result for processing (assuming single image input)
        result = results[0]
        return self.get_boxes_as_json(result, page_number)


if __name__ == "__main__":
    # Initialize the class with the model path
    model_path = "/home/akash/ws/artifacts/HW/HW_telugu_v02_081024/HW_telugu_v02_081024_2/weights/best.pt"
    image_path = "/home/akash/ws/dataset/hand_written/finetune_data/telugu_test/images/val/24300.jpg"
    page_number = 1

    # Create an instance of the class and process the image
    yolo_extractor = YOLOJsonExtractor(model_path)
    result_json = yolo_extractor.process_image(image_path, page_number)

    # Print the resulting JSON data
    print(json.dumps(result_json, indent=4, ensure_ascii=False))
