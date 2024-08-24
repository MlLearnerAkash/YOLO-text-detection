# import typer
import yaml
from ultralytics import YOLO
from ultralytics import settings
import typer
import mlflow
import re
import os

os.environ['MLFLOW_TRACKING_URI'] = 'http://10.10.16.13:5000'

settings.update({
    'mlflow': True,
    'wandb' : False})



def on_fit_epoch_end(trainer):
    print('in the on_fit_epoch_end')
    metrics_dict = {f"{re.sub('[()]', '', k)}": float(v) for k, v in trainer.metrics.items()}
    mlflow.log_metrics(metrics=metrics_dict, step=trainer.epoch)

def main(
    base_model: str,
    datasets: str = "/home/akash/ws/YOLO-text-detection/ultralytics/ultralytics/cfg/datasets/custom_data.yaml",
    epochs: int = 150,
    imgsz: int = 1024,
    batch: int = 7,
    dropout: float = 0.0,
    resume: bool = False,
    device = "0",
    name: str= "hindi-fineune",
    tracking_uri: str = "http://10.10.16.13:5000"
):
    try:
        # from clearml import Task
        

        # Task.init(
        #     project_name="malayam-word-det-20082024",
        #     task_name=f"epochs-{epochs}-imgsz-{imgsz}-batch-{batch}",
        # )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(name)
        #mlflow.set_tag('mlflow.runName', 'yolov8')
    except ImportError:
        print("mlflow not initlaized")
    with mlflow.start_run():
        model = YOLO(base_model)
        model.add_callback("on_fit_epoch_end",on_fit_epoch_end)
        results = model.train(
            data=datasets,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            dropout=dropout,
            resume=resume,
            device = device,
            name= name,

        )
        # mlflow.log_params("")
        print(results)
        mlflow.end_run()


if __name__ == "__main__":
    typer.run(main)