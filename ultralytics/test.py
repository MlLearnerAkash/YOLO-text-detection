# from ultralytics import YOLO

from ultralytics import YOLO

# # Load a model
model = YOLO("/home/akash/ws/YOLO-text-detection/ultralytics/runs/detect/hindi_finetune_base_model_090924/weights/best.pt")  # pretrained YOLOv8n model
source = "/home/akash/ws/dataset/hand_written/test_data/book_851_ravi_style_gt/ravi_style_gt/images"
# Run batched inference on a list of images
results = model(source= source,
                conf=0.15, iou= 0.15, save_txt= True, save = True)  # return a list of Results objects

# Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk


