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
    checkpoint = "/root/volume/OUTPUT/MP_TRAIN_3_maximal_crop_2025-03-11_15-09-26/checkpoint-best/mp_rank_00_model_states.pt"
    output_dir = "/root/volume/results/evaluation"
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

    # Evaluate
    conf = 0.6
    all_predictions = {}
    all_ground_truth = {}
    video_lengths = {}
    for video_file in tqdm(video_files):
        video_path = os.path.join(video_dir, video_file)
        ann_path = os.path.join(ann_dir, f"{video_file}.json")
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
    evaluation_results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(evaluation_results_path, 'w') as f:
        json.dump(results, f, indent=4)