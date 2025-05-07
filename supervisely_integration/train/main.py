import os
from supervisely_integration.train.trainer import TrainAppMVD
from run_class_finetuning import main as finetune
from run_class_finetuning import get_args as get_finetune_args

base_path = "supervisely_integration/train"
train = TrainAppMVD(
    "MVD",
    f"supervisely_integration/models.json",
    f"{base_path}/hyperparameters.yaml",
    f"{base_path}/app_options.yaml",
)

@train.start
def start_training():
    opts, ds_init = get_train_args()
    train.start_tensorboard(train.log_dir)
    finetune(opts, ds_init)
    
    experiment_info = {
        "model_name": train.model_name,
        "model_files": {},
        "checkpoints": train.output_dir,
        "best_checkpoint": "best.pth",
    }
    return experiment_info

def get_train_args():
    project = train.sly_project
    train_dataset = project.datasets.get("train")
    if train_dataset is None:
        raise ValueError("Dataset: 'train' not found. Make sure you are using the correct project.")

    # Init Default
    opts, ds_init = get_finetune_args()
    # Static
    opts.model = "vit_small_patch16_224"
    opts.data_set = "Kinetics-400"
    opts.finetune = train.model_files["checkpoint"]
    opts.nb_classes = "3"
    opts.data_path = train_dataset.directory
    opts.data_root = train_dataset.directory
    opts.det_anno_path = os.path.join(train_dataset.directory, "datasets")
    opts.log_dir = train.log_dir
    opts.output_dir = train.output_dir
    # Hyperparameters
    for key, value in train.hyperparameters.items():
        setattr(opts, key, value)
    return opts, ds_init
