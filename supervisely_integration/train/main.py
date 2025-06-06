import os
from supervisely_integration.train.trainer import TrainAppMVD
from supervisely_integration.train.scripts.training import finetune, get_train_args

if os.environ.get("LOGLEVEL", "").lower() == "info":
    os.environ["LOGLEVEL"] = "INFO"
# Do not remove the imports!
import deepspeed
from deepspeed import DeepSpeedConfig
from mpi4py import MPI

base_path = "supervisely_integration/train"
train = TrainAppMVD(
    "MVD",
    f"supervisely_integration/models.json",
    f"{base_path}/hyperparameters.yaml",
    f"{base_path}/app_options.yaml",
)


@train.start
def start_training():
    opts, ds_init = get_train_args(
        train.sly_project,
        train.model_files["checkpoint"],
        train.hyperparameters,
        train.log_dir,
        train.output_dir,
    )
    
    train.start_tensorboard(train.log_dir)
    finetune(opts, ds_init)

    experiment_info = {
        "model_name": train.model_name,
        "model_files": {"config": os.path.join(train.output_dir, "config.txt")},
        "checkpoints": os.path.join(train.output_dir, "checkpoints"),
        "best_checkpoint": "best.pth",
    }
    return experiment_info
