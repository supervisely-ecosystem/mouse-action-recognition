import json
from pathlib import Path
import os

import numpy as np
from supervisely import VideoProject, OpenMode, VideoDataset

from src.benchmark.benchmark import (
    evaluate_frame_level,
    load_ground_truth,
    load_predictions,
)
from mouse_scripts.video_utils import get_total_frames


if __name__ == "__main__":
    class_names = ["idle", "Self-Grooming", "Head/Body TWITCH"]
    conf = 0.6

    gt_path = "/gt"
    gt_dir_name = Path(os.environ.get('GT')).name
    project = VideoProject(gt_path, mode=OpenMode.READ)

    pred_path = "/pred"

    output_path = "/output"
    benchmark_dir = Path(output_path) / Path("benchmark")

    all_predictions = {}
    all_ground_truth = {}
    video_lengths = {}

    for dataset in project.datasets:
        dataset: VideoDataset
        for video_name, video_path, ann_path in dataset.items():
            predictions_path = Path(pred_path) / Path(dataset.path) / Path(f"{video_name}.json")
            if not predictions_path.exists():
                continue

            benchmark_results_path = Path(benchmark_dir) / Path(gt_dir_name) / Path(dataset.path) / Path(f"{video_name}.json")

            predictions = load_predictions(predictions_path, class_names, conf=conf)
            ground_truth = load_ground_truth(ann_path)
            
            print(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth segments")

            # Evaluate frame-level metrics
            num_frames = get_total_frames(video_path)
            print(f"Total frames in video: {num_frames}")
            frame_level_results = evaluate_frame_level(predictions, ground_truth, num_frames, class_names[1:])
            print("\n=== Frame Level Evaluation ===")
            print(frame_level_results)
    
            data = {
                cls_name: {k:int(v) if isinstance(v, np.int64) else v for k, v in metrics.items()}
                for cls_name, metrics in frame_level_results.items()
            }

            os.makedirs(str(benchmark_results_path.parent), exist_ok=True)
            json.dump(data, open(benchmark_results_path, "w"), indent=4)

            print(f"\nMetrcis data for {video_name} saved to {benchmark_results_path}\n")

            # For aggregated metrics
            all_predictions[video_path] = predictions
            all_ground_truth[video_path] = ground_truth
            video_lengths[video_path] = num_frames

    # Evaluate frame-level metrics
    from src.benchmark.benchmark import evaluate_dataset_micro_average
    results = evaluate_dataset_micro_average(
        all_predictions,
        all_ground_truth,
        video_lengths,
        class_names,
    )
    results = {
        cls_name: {k:int(v) if isinstance(v, np.int64) else v for k, v in metrics.items()}
        for cls_name, metrics in results.items()
    }
    print("Evaluation Results:")
    print(results)
    
    # Save evaluation results
    evaluation_results_path = os.path.join(str(benchmark_dir), "aggregated_results.json")
    with open(evaluation_results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\Aggregated metrics saved to {benchmark_results_path}\n")