# pipeline/semi_supervised_pipeline.py
import os
from ultralytics import YOLO
import argparse



# Argument parser for pipeline configuration
def get_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Semi-Supervised Learning Pipeline")
    parser.add_argument("--super_model_path", type= str, help= "super-vised model path")
    parser.add_argument('--epochs_supervised', type=int, default=50, help="Number of epochs for supervised learning")
    parser.add_argument('--epochs_semi', type=int, default=100, help="Number of epochs for semi-supervised learning")
    parser.add_argument('--imgsz', type=int, default=1024, help="Image size for training and inference")
    parser.add_argument('--conf_thresh', type=float, default=0.5, help="Confidence threshold for pseudo-labeling")
    parser.add_argument('--unlabeled_dir', type=str, default='datasets/unlabeled/images', help="Directory for unlabeled images")
    parser.add_argument('--labeled_dir', type=str, default='datasets/labeled/images', help="Directory for labeled images")


    return parser.parse_args()


# Pseudo-labeling the unlabeled dataset
def generate_pseudo_labels(model, unlabeled_dir, conf_thresh, imgsz):
    print(f"Generating pseudo-labels for the unlabeled dataset with confidence threshold {conf_thresh}...")
    results = model.predict(source=unlabeled_dir, save_txt=True,
                            save_conf=False, conf=conf_thresh, 
                            imgsz=imgsz, save = False, device ="0",
                            batch=6, name = "telugu"
                            )
    print("Pseudo-labels generated successfully.")




if __name__ == "__main__":
    args = get_args()

    # Load pretrained YOLOv8 model
    model = YOLO(args.super_model_path)  # Use the pretrained YOLOv8 model (replace 'n' with appropriate version)

    # Generate pseudo-labels on unlabeled data
    generate_pseudo_labels(model, unlabeled_dir=args.unlabeled_dir, conf_thresh=args.conf_thresh, 
                           imgsz=args.imgsz)