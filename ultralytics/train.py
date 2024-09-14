# import typer
import yaml
from ultralytics import YOLO
from ultralytics import settings
import typer
import mlflow
import re
import os
import dagshub


try:
    os.environ['MLFLOW_TRACKING_URI'] = 'http://10.10.16.13:5000'
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "HW_hindi_130924"
    # mlflow.set_tag('mlflow.runName', 'yolov8')
    dagshub.init(repo_owner='manna.phys', repo_name='YOLO-text-detection', mlflow=True)
except ImportError:
    print("mlflow not initlaized")


settings.update({
    'mlflow': True,
    'wandb' : False})



def on_fit_epoch_end(trainer):
    print('in the on_fit_epoch_end')
    metrics_dict = {f"{re.sub('[()]', '', k)}": float(v) for k, v in trainer.metrics.items()}
    mlflow.log_metrics(metrics=metrics_dict, step=trainer.epoch)

def main(
    base_model: str,
    datasets: str = "/home/akash/ws/YOLO-text-detection/ultralytics/ultralytics/cfg/datasets/hindi.yaml",
    epochs: int = 250,
    imgsz: int = 1024,
    batch: int = 6,
    dropout: float = 0.0,
    resume: bool = False,
    device = "0",
    name: str= "HW_hindi_130924_",
    project = "//home/akash/ws/artifacts/HW/hindi_230924"
):
    
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
            project = project

        )
        # mlflow.log_params("")
        # print(results)
        mlflow.end_run()


if __name__ == "__main__":
    typer.run(main)