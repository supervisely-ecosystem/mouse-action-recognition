import json
import os
from pathlib import Path
import decord
import numpy as np

from tqdm import tqdm
from src.inference.visualize import draw_timeline, write_positive_fragments
from src.inference.inference import predict_video_with_detector, load_mvd, load_detector, postprocess_predictions


if __name__ == '__main__':
    checkpoint = "/root/volume/OUTPUT/MP_TRAIN_3_maximal_crop_2025-03-11_15-09-26/checkpoint-best/mp_rank_00_model_states.pt"
    video_path = "/root/volume/data/mouse/HOM Mice F.2632_HOM 12 Days post tre/12 Days post tre/video/GL010560.MP4"
    ann_path = Path(video_path).parent.parent / "ann" / (Path(video_path).name + ".json")
    STRIDE = 8  # 8x2=16 (16 stride, 32 context window)
    detector_url = "http://supervisely-utils-rtdetrv2-inference-1:8000"

    experiment_name = checkpoint.split('/')[-3]
    print(f"Experiment name: {experiment_name}")
    checkpoint = Path(checkpoint)
    assert checkpoint.exists(), f"Checkpoint {checkpoint} does not exist."
    assert Path(video_path).exists(), f"Video {video_path} does not exist."
    assert ann_path.exists(), f"Annotation {ann_path} does not exist."
    output_dir = checkpoint.parent.parent
    
    # Load models
    model, opts = load_mvd(checkpoint)
    detector = load_detector(session_url=detector_url)

    # Predict
    predictions = predict_video_with_detector(
        video_path,
        model,
        detector,
        opts,
        stride=STRIDE
    )
    predictions = postprocess_predictions(predictions)

    # Save predictions to JSON file
    os.makedirs("results2", exist_ok=True)
    output_json_path = f"results2/predictions_{experiment_name}.json"
    with open(output_json_path, 'w') as f:
        json.dump(predictions, f, indent=4)
        
    class_names = ["idle", "Self-Grooming", "Head/Body Twitch"]
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()    
    draw_timeline(predictions, fps, experiment_name=experiment_name, class_names=class_names,
                 figsize=(15, 7))
    
    # Display additional statistics
    # Extract probabilities from the predictions list
    all_probs = np.array([pred['probabilities'] for pred in predictions])
    num_windows = len(predictions)
    
    # Get the most probable class for each window
    dominant_classes = np.argmax(all_probs, axis=1)
    class_counts = np.bincount(dominant_classes)
    top_classes = np.argsort(-class_counts)[:5]
    print("\nDominant classes in the video:")
    for cls in top_classes:
        if class_counts[cls] > 0:
            percentage = (class_counts[cls] / num_windows) * 100
            print(f"Class {cls}: {percentage:.2f}% ({class_counts[cls]} windows)")

    write_positive_fragments(predictions, video_path, crop=True, output_dir="results2")
