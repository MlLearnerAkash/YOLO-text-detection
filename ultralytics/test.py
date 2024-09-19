# from ultralytics import YOLO

from ultralytics import YOLO

# # Load a model
model = YOLO("/home/akash/ws/artifacts/HW/oriya_170924/HW_oriya_170924_/weights/best.pt")  # pretrained YOLOv8n model
source = "/home/akash/ws/dataset/hand_written/test_data/oriya/book_901_ravi_style_gt/ravi_style_gt/images"
# Run batched inference on a list of images
results = model(source= source,
                conf=0.15, iou= 0.15, save_txt= True, save = True,
                line_width= 1,
                imgsz = (1024, 1024))  # return a list of Results objects

# Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk


