# import typer
import yaml
from ultralytics import YOLO
from ultralytics import settings
import typer
import mlflow
import re
import os

os.environ['MLFLOW_TRACKING_URI'] = 'http://10.10.16.13:5000'
os.environ["MLFLOW_EXPERIMENT_NAME"] = "hindi-finetune-val"
settings.update({
    'mlflow': False,
    'wandb' : False})



def on_fit_epoch_end(trainer):
    print('in the on_fit_epoch_end')
    metrics_dict = {f"{re.sub('[()]', '', k)}": float(v) for k, v in trainer.metrics.items()}
    mlflow.log_metrics(metrics=metrics_dict, step=trainer.epoch)

def main(
    base_model: str,
    datasets: str = "/home/akash/ws/YOLO-text-detection/ultralytics/ultralytics/cfg/datasets/hindi-finetune-dataset.yaml",
    imgsz: int = 1024,
    batch: int = 7,
    device = "0",
    conf: float = 0.55,
    iou: float = 0.25,
    name: str= "hindi-finetune-val-97-train-810-test",
    tracking_uri: str = "http://10.10.16.13:5000"
):
    try:
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(name)
        # mlflow.set_tag('mlflow.runName', 'yolov8')
    except ImportError:
        print("mlflow not initlaized")
    # with mlflow.start_run():
    model = YOLO(base_model)
    # model.add_callback("on_fit_epoch_end",on_fit_epoch_end)
    results = model.val(
        data=datasets,
        imgsz=imgsz,
        batch=batch,
        device = device,
        name= name,
        conf = conf,
        iou = iou


    )
    # mlflow.log_params("")
    # print(results)
    # mlflow.end_run()


if __name__ == "__main__":
    typer.run(main)