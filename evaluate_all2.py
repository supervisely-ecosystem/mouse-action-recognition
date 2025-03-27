import json
from pathlib import Path
import os

from tqdm import tqdm
from src.benchmark.benchmark import evaluate_dataset_micro_average, load_ground_truth, load_predictions
from mouse_scripts.video_utils import get_total_frames
from src.inference.inference import predict_video_with_detector, load_mvd, load_detector, postprocess_predictions


if __name__ == "__main__":
    CLASS_NAMES = ["idle", "Self-Grooming", "Head/Body TWITCH"]
    dataset_dir = "/root/volume/data/mouse/sampled_dataset"
    predictions_dir = "/root/volume/results/evaluation/MP_TRAIN_3_maximal_crop_2025-03-11_15-09-26/predictions"
    evaluation_results_path = Path(predictions_dir).parent / "evaluation_results_fixed.json"

    # Get list of video files
    video_dir = os.path.join(dataset_dir, "videos")
    ann_dir = os.path.join(dataset_dir, "annotations")
    video_files = sorted([f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')])
    print(f"Found {len(video_files)} video files in {video_dir}")

    # Evaluate
    conf = 0.6
    all_predictions = {}
    all_ground_truth = {}
    video_lengths = {}
    for video_file in tqdm(video_files):
        video_path = os.path.join(video_dir, video_file)
        ann_path = os.path.join(ann_dir, f"{video_file}.json")
        if not os.path.exists(ann_path):
            print(f"Annotation file not found for {video_file}, skipping...")
            continue
        predictions_path = os.path.join(predictions_dir, f"{video_file}.json")
        predictions = load_predictions(predictions_path, CLASS_NAMES, conf=conf)
        ground_truth = load_ground_truth(ann_path)
        all_predictions[video_file] = predictions
        all_ground_truth[video_file] = ground_truth
        video_lengths[video_file] = get_total_frames(video_path)
    
    # Evaluate frame-level metrics
    from src.benchmark.benchmark import evaluate_dataset_micro_average
    results = evaluate_dataset_micro_average(
        all_predictions,
        all_ground_truth,
        video_lengths,
        CLASS_NAMES,
    )
    print("Evaluation Results:")
    print(results)
    
    # Save evaluation results
    with open(evaluation_results_path, 'w') as f:
        json.dump(results, f, indent=4)