import os
import csv
import json
import random
import numpy as np
from pathlib import Path
from supervisely import (
    logger,
    VideoDataset,
    VideoProject,
    OpenMode,
    ProjectInfo,
    VideoAnnotation,
)
from supervisely.nn.training.train_app import TrainApp
import supervisely.io.fs as sly_fs
import inference_script.run_benchmark as mvd_benchmark
from supervisely_integration.train.scripts.inference import run_inference
from supervisely.io.json import load_json_file


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
        train_csv_path = os.path.join(datasets_path, "train.csv")
        val_csv_path = os.path.join(datasets_path, "val.csv")

        all_videos = []
        for category, label in categories.items():
            category_video_dir = os.path.join(datasets_path, category, "video")
            if os.path.exists(category_video_dir):
                video_files = [
                    f
                    for f in os.listdir(category_video_dir)
                    if f.endswith((".mp4", ".MP4"))
                ]
                for video_file in video_files:
                    video_path = f"{category}/video/{video_file}"
                    all_videos.append((video_path, label))

        random.shuffle(all_videos)
        split_idx = int(len(all_videos) * (1 - val_ratio))
        train_videos, val_videos = all_videos[:split_idx], all_videos[split_idx:]
        self._train_split, self._val_split = train_videos, val_videos
        with open(train_csv_path, "w", newline="") as train_file:
            writer = csv.writer(train_file, delimiter=" ")
            for video_path, label in train_videos:
                writer.writerow([video_path, label])
        with open(val_csv_path, "w", newline="") as val_file:
            writer = csv.writer(val_file, delimiter=" ")
            for video_path, label in val_videos:
                writer.writerow([video_path, label])
        logger.info(
            f"Created {len(train_videos)} records in train.csv and {len(val_videos)} records in val.csv"
        )

    def _run_model_benchmark(
        self,
        local_artifacts_dir: str,
        remote_artifacts_dir: str,
        experiment_info: dict,
        splits_data: dict,
        model_meta: ProjectInfo,
        gt_project_id: int = None,
    ):
        lnk_file_info, report, report_id, eval_metrics, primary_metric_name = (
            None,
            None,
            None,
            {},
            None,
        )
        # return lnk_file_info, report, report_id, eval_metrics, primary_metric_name

        best_checkpoint = experiment_info.get("best_checkpoint", None)
        best_filename = sly_fs.get_file_name_with_ext(best_checkpoint)
        logger.info(f"Creating the report for the best model: {best_filename!r}")

        # TODO:
        # инферить только мвд на тест видео -> pred_dir
        config_path = os.path.join(local_artifacts_dir, "config.txt")
        video_dir = os.path.join(self.sly_project.directory, "test", "video")
        pred_dir = os.path.join(self.work_dir, "predictions") # sly_fs.mkdir(pred_dir, True)
        run_inference(video_dir, pred_dir, model_meta, best_checkpoint, config_path)

        tag_names = [
            "idle",
            "Self-Grooming",
            "Head/Body TWITCH",
        ]  # check correct tag names after training
        conf = 0.6

        gt_path = self.sly_project.directory
        gt_dir_name = "sly_project"
        project = VideoProject(gt_path, mode=OpenMode.READ)

        # папка куда модель сохраняет предикты

        output_path = "./output"
        benchmark_dir = Path(output_path) / Path("benchmark")

        all_predictions = {}
        all_ground_truth = {}
        video_lengths = {}
        all_results = {}

        for dataset in project.datasets:
            dataset: VideoDataset
            for video_name, video_path, ann_path in dataset.items():
                predictions_path = Path(pred_dir) / Path(f"{video_name}.json")
                if not predictions_path.exists():
                    raise FileNotFoundError(
                        f"Predictions file not found: {predictions_path}"
                    )

                benchmark_results_path = (
                    Path(benchmark_dir)
                    / Path(gt_dir_name)
                    / Path(dataset.path)
                    / Path(f"{video_name}.json")
                )

                predictions = mvd_benchmark.load_predictions(
                    predictions_path, tag_names, conf=conf
                )
                ground_truth = mvd_benchmark.load_ground_truth(ann_path)

                print(
                    f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth segments"
                )

                # Evaluate frame-level metrics
                num_frames = mvd_benchmark.get_total_frames(video_path)
                print(f"Total frames in video: {num_frames}")
                frame_level_results = mvd_benchmark.evaluate_frame_level(
                    predictions, ground_truth, num_frames, tag_names[1:]
                )
                print("\n=== Frame Level Evaluation ===")
                print(frame_level_results)

                data = {
                    cls_name: {
                        k: int(v) if isinstance(v, np.int64) else v
                        for k, v in metrics.items()
                    }
                    for cls_name, metrics in frame_level_results.items()
                }
                video_key = (
                    video_path.replace("/datasets/", "/")
                    .replace("/video/", "/")
                    .replace(gt_path, "")
                    .lstrip("/")
                )
                all_results[video_key] = data

                os.makedirs(str(benchmark_results_path.parent), exist_ok=True)
                json.dump(data, open(benchmark_results_path, "w"), indent=4)

                print(
                    f"\nMetrcis data for {video_name} saved to {benchmark_results_path}\n"
                )

                # For aggregated metrics
                all_predictions[video_key] = predictions
                all_ground_truth[video_key] = ground_truth
                video_lengths[video_key] = num_frames

        # Evaluate frame-level metrics
        from src.benchmark.benchmark import evaluate_dataset_micro_average

        results = evaluate_dataset_micro_average(
            all_predictions,
            all_ground_truth,
            video_lengths,
            tag_names,
        )
        all_results["aggregated"] = {
            cls_name: {
                k: int(v) if isinstance(v, np.int64) else v for k, v in metrics.items()
            }
            for cls_name, metrics in results.items()
        }

        # Save evaluation results
        evaluation_results_path = os.path.join(
            str(benchmark_dir), "aggregated_results.json"
        )
        with open(evaluation_results_path, "w") as f:
            json.dump(all_results["aggregated"], f, indent=4)
        print(f"Aggregated metrics saved to {benchmark_results_path}\n")

        mvd_benchmark.visuailize(benchmark_dir, all_results)
        return lnk_file_info, report, report_id, eval_metrics, primary_metric_name
