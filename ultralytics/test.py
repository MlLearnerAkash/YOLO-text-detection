# from ultralytics import YOLO

from ultralytics import YOLO
import json

import json

def save_boxes_to_json(result, page_number, output_file):
    """
    Save YOLO bounding boxes to JSON format without text or reading order.

    Args:
        result: YOLO model result object containing boxes.
        page_number (int): The page number corresponding to the detection.
        output_file (str): File path to save the output JSON.
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
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"Bounding box data saved to {output_file}")


if __name__ == "__main__":
    # # Load a model
    model = YOLO("/home/akash/ws/artifacts/HW/HW_telugu_v02_081024/HW_telugu_v02_081024_2/weights/best.pt")  # pretrained YOLOv8n model
    source = "/home/akash/ws/dataset/hand_written/finetune_data/telugu_test/images/val/24300.jpg"
    # Run batched inference on a list of images
    results = model(source= source,
                    conf=0.15, iou= 0.15, save_txt= True, save = False,
                    save_json = True,
                    line_width= 1,
                    imgsz = (1024, 1024),
                    project= "./relaxed_recall",
                    name = "yolo_result")  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs

    result = results[0]  # Placeholder; use the actual YOLO result here
    page_number = 1
    output_file = "output.json"

    # Uncomment and populate `result` with actual data before running
    save_boxes_to_json(result, page_number, output_file)



