import os
from autodistill_dinov2 import DINOv2
from autodistill.detection import CaptionOntology

base_model = DINOv2(ontology=CaptionOntology({"shipping container": "container"}))

IMAGE_NAME = "/home/akash/ws/YOLO-text-detection/runs/detect/predict/52441.jpg"
DATASET_NAME = "test"
image = os.path.join(DATASET_NAME, IMAGE_NAME)

predictions = base_model.predict(image)