import os
import csv
import random
from supervisely import logger, VideoDataset
from supervisely.nn.training.train_app import TrainApp
import supervisely.io.fs as sly_fs

class TrainAppMVD(TrainApp):
    def __init__(self, model_name, model_files, hyperparameters, app_options):
        super().__init__(model_name, model_files, hyperparameters, app_options)

    @property
    def classes(self):
        return ["mouse"]
    
    @property
    def tags(self):
        return ["idle", "Head-Body_TWITCH", "Self-Grooming"]

    # Debug without downloading project | Remove later
    def _prepare_working_dir(self):
        # sly_fs.mkdir(self.work_dir, True)
        sly_fs.mkdir(self.output_dir, True)
        sly_fs.mkdir(self._output_checkpoints_dir, True)
        # sly_fs.mkdir(self.project_dir, True)
        sly_fs.mkdir(self.model_dir, True)
        sly_fs.mkdir(self.log_dir, True)

    def _download_project(self):
        self._read_project()
    # -------------------------------- #

    def _split_project(self):
        val_ratio = self.hyperparameters["val_ratio"]
        train_dataset: VideoDataset = self.sly_project.datasets.get("train")
        # test_dataset: VideoDataset = self.sly_project.datasets.get("test")

        categories = {"idle": 0, "Head-Body_TWITCH": 1, "Self-Grooming": 2}
        datasets_path = os.path.join(train_dataset.directory, "datasets")
        csv_dir = train_dataset.directory
        os.makedirs(csv_dir, exist_ok=True)
        train_csv_path = os.path.join(csv_dir, "train.csv")
        val_csv_path = os.path.join(csv_dir, "val.csv")
        
        all_videos = []
        for category, label in categories.items():
            category_video_dir = os.path.join(datasets_path, category, "video")
            if os.path.exists(category_video_dir):
                video_files = [f for f in os.listdir(category_video_dir) if f.endswith(('.mp4', '.MP4'))]
                for video_file in video_files:
                    video_path = f"{datasets_path}/{category}/video/{video_file}"
                    all_videos.append((video_path, label))
        
        random.shuffle(all_videos)
        split_idx = int(len(all_videos) * (1 - val_ratio))
        train_videos, val_videos = all_videos[:split_idx], all_videos[split_idx:]
        self._train_split, self._val_split = train_videos, val_videos
        with open(train_csv_path, 'w', newline='') as train_file:
            writer = csv.writer(train_file, delimiter=' ')
            for video_path, label in train_videos:
                writer.writerow([video_path, label])
        with open(val_csv_path, 'w', newline='') as val_file:
            writer = csv.writer(val_file, delimiter=' ')
            for video_path, label in val_videos:
                writer.writerow([video_path, label])
        logger.info(f"Created {len(train_videos)} records in train.csv and {len(val_videos)} records in val.csv")

    def _run_model_benchmark(self):
        pass