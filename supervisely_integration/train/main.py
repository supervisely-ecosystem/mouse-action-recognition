import os
import torch
import yaml

import supervisely as sly
from supervisely.nn import ModelSource, RuntimeType
from supervisely.nn.training.train_app import TrainApp
# from supervisely_integration.train.trainer import TrainAppMVD



base_path = "supervisely_integration/train"
train = TrainApp(
    "MVD",
    f"supervisely_integration/models.json",
    f"{base_path}/hyperparameters.yaml",
    f"{base_path}/app_options.yaml",
)



@train.start
def start_training():
    train_ann_path, val_ann_path = get_datasets()
    checkpoint = train.model_files["checkpoint"]
    # gather experiment info
    experiment_info = {
        "model_name": train.model_name,
        # "model_files": {"config": model_config_path},
        # "checkpoints": output_dir,
        "best_checkpoint": "best.pth",
    }
    return experiment_info


def get_datasets():
    project = train.sly_project
    meta = project.meta

    # Get Datasets
    # Test
    test_dataset: sly.VideoDataset = project.datasets.get("test")
    # Train
    train_dataset: sly.VideoDataset = project.datasets.get("train")
    sg_dataset: sly.VideoDataset = project.datasets.get("train/Self-Grooming")
    hbt_dataset: sly.VideoDataset = project.datasets.get("train/Head-Body_TWITCH")
    idle_dataset: sly.VideoDataset = project.datasets.get("train/idle")

    train_dataset.path

def get_train_args():
    train_args = []
    train_args.append("--nb_classes")
    train_args.append("3")
    train_args.append("--data_path")
    train_args.append("/root/volume/data/mouse/")
    train_args.append("--data_root")
    train_args.append("/root/volume/data/mouse/")
