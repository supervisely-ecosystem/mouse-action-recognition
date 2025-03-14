import json
from pathlib import Path
import os

from tqdm import tqdm
from src.benchmark.benchmark import evaluate_dataset_micro_average, load_ground_truth, load_predictions
from mouse_scripts.video_utils import get_total_frames
from src.inference.inference import predict_video_with_detector, load_mvd, load_detector, postprocess_predictions


if __name__ == "__main__":
    CLASS_NAMES = ["idle", "Self-Grooming", "Head/Body TWITCH"]
    dataset_dir = "data/mouse/sampled_dataset"
    checkpoint = "/root/volume/OUTPUT/MP_TRAIN_3_maximal_crop_2025-03-11_15-09-26/checkpoint-best/mp_rank_00_model_states.pt"
    output_dir = "results/evaluation"
    detector_url = "http://supervisely-utils-rtdetrv2-inference-1:8000"

    experiment_name = checkpoint.split('/')[-3]
    output_dir = f"{output_dir}/{experiment_name}"
    predictions_dir = f"{output_dir}/predictions"
    os.makedirs(output_dir, exist_ok=False)
    os.makedirs(predictions_dir)

    # Load models
    model, opts = load_mvd(checkpoint)
    detector = load_detector(detector_url)

    # Get list of video files
    video_dir = os.path.join(dataset_dir, "videos")
    ann_dir = os.path.join(dataset_dir, "annotations")
    video_files = sorted([f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')])
    print(f"Found {len(video_files)} video files in {video_dir}")

    # Inference    
    for video_file in tqdm(video_files):
        video_path = os.path.join(video_dir, video_file)
        ann_path = os.path.join(ann_dir, f"{video_file}.json")

        predictions = predict_video_with_detector(
            video_path,
            model,
            detector,
            opts,
            stride=8
        )
        predictions = postprocess_predictions(predictions)
        
        # Save predictions to JSON file
        output_json_path = f"{predictions_dir}/{video_file}.json"
        with open(output_json_path, 'w') as f:
            json.dump(predictions, f, indent=4)
