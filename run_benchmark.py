import json
from pathlib import Path
from typing import List
from src.inference.inference import postprocess_predictions
from src.benchmark.benchmark import TemporalMetrics, TemporalMatcher, ActionSegment, evaluate_frame_level, load_ground_truth
from mouse_scripts.video_utils import get_total_frames


def load_predictions(predictions_path: str, class_names: list, conf=None) -> List[ActionSegment]:
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)

    predictions = postprocess_predictions(predictions, conf)
    segments = [ActionSegment(
        start_frame=pred['frame_range'][0],
        end_frame=pred['frame_range'][1],
        action_class=class_names[pred['label']],
        confidence=pred['confidence'],
    ) for pred in predictions]
    return segments


if __name__ == "__main__":
    class_names = ["idle", "Self-Grooming", "Head/Body TWITCH"]

    video_path = "/root/volume/data/mouse/HOM Mice F.2632_HOM 12 Days post tre/12 Days post tre/video/GL010560.MP4"
    ann_path = Path(video_path).parent.parent / "ann" / (Path(video_path).name + ".json")
    predictions_path = "results/predictions_MP_TRAIN_3_maximal_crop_2025-03-11_15-09-26.json"
        
    predictions = load_predictions(predictions_path, class_names, conf=0.7)
    ground_truth = load_ground_truth(ann_path)
    print(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth segments")

    # from random import shuffle
    # shuffle(predictions)
    # shuffle(ground_truth)

    # from benchmark_visualizations import draw_segments
    # draw_segments(predictions, ground_truth, "Self-Grooming")

    # Evaluate frame-level metrics
    num_frames = get_total_frames(video_path)
    print(f"Total frames in video: {num_frames}")
    frame_level_results = evaluate_frame_level(predictions, ground_truth, num_frames, class_names[1:])
    print("\n=== Frame Level Evaluation ===")
    print(frame_level_results)
    exit(0)
    
    # Create matcher and find matches
    matcher = TemporalMatcher(iou_threshold=0.5)
    matches, unmatched_preds, unmatched_gt = matcher.match_segments(predictions, ground_truth)
    
    print("=== Matching Results ===")
    print(f"Found {len(matches)} matches at IoU threshold 0.5")
    # for match in matches:
    #     pred = predictions[match.pred_idx]
    #     gt = ground_truth[match.gt_idx]
    #     print(f"Match: {pred} -> {gt} (IoU: {match.iou:.2f})")
    
    # print(f"\nUnmatched predictions: {len(unmatched_preds)}")
    # for idx in unmatched_preds:
    #     print(f"  {predictions[idx]}")
    
    # print(f"\nUnmatched ground truth: {len(unmatched_gt)}")
    # for idx in unmatched_gt:
    #     print(f"  {ground_truth[idx]}")
    
    # Calculate metrics
    metrics = TemporalMetrics(iou_thresholds=[0.3, 0.5, 0.7])
    results = metrics.evaluate(predictions, ground_truth)
    
    print("\n=== Metrics ===")
    print(f"mAP: {results['mAP']:.4f}")
    
    for threshold in [0.3, 0.5, 0.7]:
        print(f"\nAt IoU={threshold}:")
        print(f"  AP: {results[f'AP@{threshold}']:.4f}")
        print(f"  Precision: {results[f'Precision@{threshold}']:.4f}")
        print(f"  Recall: {results[f'Recall@{threshold}']:.4f}")
        print(f"  F1: {results[f'F1@{threshold}']:.4f}")
    
    # Print class metrics at IoU=0.5
    print("\n=== Class Metrics (IoU=0.5) ===")
    class_metrics = results["class_metrics@0.5"]
    for class_name, metrics in class_metrics.items():
        print(f"\n{class_name.upper()}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")