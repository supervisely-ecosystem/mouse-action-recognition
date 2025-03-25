import json
from pathlib import Path
import os

from supervisely import VideoProject, OpenMode, VideoDataset

from src.benchmark.benchmark import (
    evaluate_frame_level,
    load_ground_truth,
    load_predictions,
)
from mouse_scripts.video_utils import get_total_frames


if __name__ == "__main__":
    class_names = ["idle", "Self-Grooming", "Head/Body TWITCH"]

    input_path = "/input"
    input_project_name = Path(os.environ.get('INPUT')).name
    project = VideoProject(input_path, mode=OpenMode.READ)
    input_project_name = Path(os.environ.get('INPUT')).name

    output_path = "/output"
    output_predictions_path = Path(output_path) / Path(input_project_name) / Path("predictions")
    benchmark_dir = Path(output_path) / Path("benchmark")
    
    for dataset in project.datasets:
        dataset: VideoDataset
        for video_name, video_path, ann_path in dataset.items():
            predictions_path = output_predictions_path / Path(f"predictions_{video_name}.json")
            benchmark_results_path = Path(benchmark_dir) / Path(input_project_name) / Path(dataset.path) / Path(f"{video_name}.json")

            predictions = load_predictions(predictions_path, class_names, conf=0.7)
            ground_truth = load_ground_truth(ann_path)
            
            print(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth segments")

            # Evaluate frame-level metrics
            num_frames = get_total_frames(video_path)
            print(f"Total frames in video: {num_frames}")
            frame_level_results = evaluate_frame_level(predictions, ground_truth, num_frames, class_names[1:])
            print("\n=== Frame Level Evaluation ===")
            print(frame_level_results)
    
            data = frame_level_results

            os.makedirs(str(benchmark_results_path.parent), exist_ok=True)
            json.dump(data, open(benchmark_results_path, "w"), indent=4)

            print(f"\nMetrcis data for {video_name} saved to {benchmark_results_path}\n")
