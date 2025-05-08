import os
import csv
import json
import random
import numpy as np
from pathlib import Path
from supervisely import logger, VideoDataset, VideoProject, OpenMode
from supervisely.nn.training.train_app import TrainApp
import supervisely.io.fs as sly_fs
# import inference_script.run_benchmark as mvd_benchmark


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
                video_files = [f for f in os.listdir(category_video_dir) if f.endswith(('.mp4', '.MP4'))]
                for video_file in video_files:
                    video_path = f"{category}/video/{video_file}"
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

    # def run_inference(self):
    #     class_names = ["idle", "Self-Grooming", "Head/Body TWITCH"]
    #     detector_container_port = 8000
    #     detector_url = f"http://rt-detr:{detector_container_port}"
        
    #     model_dir = os.getenv("MODEL_DIR")
    #     checkpoint = f"{model_dir}/checkpoint-best/mp_rank_00_model_states.pt"
    #     input_path = "/input"
    #     output_dir = "/output"
        
    #     input_dir_name = Path(os.environ.get('INPUT')).name

    #     is_directory = False
    #     is_project = False
    #     project = None
    #     if os.path.isdir(input_path):
    #         is_directory = True
    #         try:
    #             project = VideoProject(input_path, mode=OpenMode.READ)
    #             is_project = True
    #         except:
    #             pass

    #     # Load models
    #     model, opts = load_mvd(checkpoint)
    #     # detector = load_detector(session_url=detector_url)

    #     if is_project:
    #         print("Predicting project")
    #         inference_project(project, input_dir_name, output_dir, class_names, model, opts, detector)
    #     elif is_directory:
    #         print("Predicting directory")
    #         inference_directory(input_path, input_dir_name, output_dir, class_names, model, opts, detector)
    #     else:
    #         print("Predicting video")
    #         input_video_name = os.path.splitext(input_dir_name)[0]
    #         output_predictions_path = Path(output_dir) / Path(input_video_name) / Path("predictions")
    #         output_project_path = Path(output_dir) / Path(input_video_name) / Path("SLY_project")
    #         output_project = VideoProject(str(output_project_path), mode=OpenMode.CREATE)
    #         output_meta = create_meta(class_names)
    #         output_project.set_meta(output_meta)
    #         output_dataset = output_project.create_dataset("predictions")
    #         inference_video(input_path, output_predictions_path, output_dataset, output_meta, class_names, model, opts, detector, input_video_name)

    def _run_model_benchmark(self):
        lnk_file_info, report, report_id, eval_metrics, primary_metric_name = None, None, None, {}, None
        return lnk_file_info, report, report_id, eval_metrics, primary_metric_name

        # TODO:
        # pred_path = self.run_inference()


        # tag_names = ["idle", "Self-Grooming", "Head/Body TWITCH"] # check correct tag names after training
        # conf = 0.6

        # gt_path = self.sly_project.directory
        # gt_dir_name = "sly_project"
        # project = VideoProject(gt_path, mode=OpenMode.READ)

        # # папка куда модель сохраняет предикты
        # pred_path = "/root/volume/results/evaluation/MP_TRAIN_3_maximal_crop_2025-03-11_15-09-26/predictions"

        # output_path = "./output"
        # benchmark_dir = Path(output_path) / Path("benchmark")

        # all_predictions = {}
        # all_ground_truth = {}
        # video_lengths = {}
        # all_results = {}

        # for dataset in project.datasets:
        #     dataset: VideoDataset
        #     for video_name, video_path, ann_path in dataset.items():
        #         predictions_path = Path(pred_path) / Path(f"{video_name}.json")
        #         if not predictions_path.exists():
        #             raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

        #         benchmark_results_path = Path(benchmark_dir) / Path(gt_dir_name) / Path(dataset.path) / Path(f"{video_name}.json")

        #         predictions = mvd_benchmark.load_predictions(predictions_path, tag_names, conf=conf)
        #         ground_truth = mvd_benchmark.load_ground_truth(ann_path)
                
        #         print(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth segments")

        #         # Evaluate frame-level metrics
        #         num_frames = mvd_benchmark.get_total_frames(video_path)
        #         print(f"Total frames in video: {num_frames}")
        #         frame_level_results = mvd_benchmark.evaluate_frame_level(predictions, ground_truth, num_frames, tag_names[1:])
        #         print("\n=== Frame Level Evaluation ===")
        #         print(frame_level_results)
        
        #         data = {
        #             cls_name: {k:int(v) if isinstance(v, np.int64) else v for k, v in metrics.items()}
        #             for cls_name, metrics in frame_level_results.items()
        #         }
        #         video_key = video_path.replace("/datasets/", "/").replace("/video/", "/").replace(gt_path, "").lstrip("/")
        #         all_results[video_key] = data

        #         os.makedirs(str(benchmark_results_path.parent), exist_ok=True)
        #         json.dump(data, open(benchmark_results_path, "w"), indent=4)

        #         print(f"\nMetrcis data for {video_name} saved to {benchmark_results_path}\n")

        #         # For aggregated metrics
        #         all_predictions[video_key] = predictions
        #         all_ground_truth[video_key] = ground_truth
        #         video_lengths[video_key] = num_frames

        # # Evaluate frame-level metrics
        # from src.benchmark.benchmark import evaluate_dataset_micro_average
        # results = evaluate_dataset_micro_average(
        #     all_predictions,
        #     all_ground_truth,
        #     video_lengths,
        #     tag_names,
        # )
        # all_results["aggregated"] = {
        #     cls_name: {k:int(v) if isinstance(v, np.int64) else v for k, v in metrics.items()}
        #     for cls_name, metrics in results.items()
        # }

        # # Save evaluation results
        # evaluation_results_path = os.path.join(str(benchmark_dir), "aggregated_results.json")
        # with open(evaluation_results_path, 'w') as f:
        #     json.dump(all_results["aggregated"], f, indent=4)
        # print(f"Aggregated metrics saved to {benchmark_results_path}\n")

        # mvd_benchmark.visuailize(benchmark_dir, all_results)

        # return lnk_file_info, report, report_id, eval_metrics, primary_metric_name


# применить этот код на всех аннотациях в тест датасете
# class asdasd(VideoSlidingWindow):
#     def __init__(self, video_path, ann_path, num_frames=16,
#                  frame_sample_rate=2, input_size=224,
#                  stride=5, bbox_padding=0.05):
#         super().__init__(video_path, num_frames, frame_sample_rate, input_size, stride)
#         # assert isinstance(detector, SessionJSON), "Detector should be an instance of SessionJSON"
#         self.detector = detector
#         self.bbox_padding = bbox_padding
#         self.detection_cache = OrderedDict()  # Cache for storing detections by frame index
#         self.max_cache_size = 128
#         self.buffer = []
#         self.frame_index = 0
#         self.frame_sample_rate = frame_sample_rate
#         # self.ann_iterator: list[frames] = []

#     def __iter__(self):
#         # for frame in video_annotation.frames
#         for detection in self.ann_iterator:
#             self.buffer.append(detection.annotation.to_json())
#             if len(self.buffer) < self.num_frames:
#                 continue

#             frame_indices = list(range(self.frame_index, self.frame_index + self.num_frames * self.frame_sample_rate, self.frame_sample_rate))
#             buffer = self.vr.get_batch(frame_indices).asnumpy()
#             w, h = buffer.shape[2], buffer.shape[1]
#             figures = [fig for frame in self.buffer for fig in frame["objects"]]
#             for fig in figures:
#                 fig['geometry'] = fig.copy()

#             x1, y1, x2, y2 = get_maximal_bbox_crop(figures, (w, h), padding=self.bbox_padding)
#             buffer = buffer[:, y1:y2, x1:x2, :]
#             buffer = self.data_transform(buffer)

#             yield buffer, frame_indices, (x1, y1, x2, y2)

#             self.frame_index += self.stride * self.frame_sample_rate
#             for _ in range(self.stride):
#                 self.buffer.pop(0)