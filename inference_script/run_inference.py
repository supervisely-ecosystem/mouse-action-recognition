from pathlib import Path
import json
import os

import traceback
import supervisely as sly
from supervisely import VideoProject, OpenMode, VideoDataset, ProjectMeta, VideoAnnotation, ObjClass
from tqdm import tqdm
import numpy as np
np.float = float

from src.inference.visualize import draw_timeline, write_positive_fragments, draw_class_segments_timeline
from src.inference.inference import predict_video_with_detector, load_mvd, load_detector, postprocess_predictions


def create_meta(class_names) -> ProjectMeta:
    tag_metas = [sly.TagMeta(class_name, sly.TagValueType.NONE) for class_name in class_names] + [sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER)]
    meta = ProjectMeta(tag_metas=tag_metas)
    return meta

def ann_from_predictions(frame_size, frames_count, predictions, project_meta: ProjectMeta, class_names):
    print("frame size:", frame_size)
    label_to_tag_meta = {i: project_meta.get_tag_meta(class_name) for i, class_name in enumerate(class_names)}
    conf_tag_meta = project_meta.get_tag_meta("confidence")
    tags = []
    for prediction in predictions:
        tag_meta = label_to_tag_meta[prediction["label"]]
        confidence = prediction["confidence"]
        frame_range = prediction["frame_range"]
        tag = sly.VideoTag(tag_meta, frame_range=frame_range)
        conf_tag = sly.VideoTag(conf_tag_meta, value=confidence, frame_range=frame_range)
        tags.extend([tag, conf_tag])
    ann = VideoAnnotation(img_size=frame_size, frames_count=frames_count, tags=sly.VideoTagCollection(tags))
    return ann

if __name__ == '__main__':
    class_names = ["idle", "Self-Grooming", "Head/Body TWITCH"]

    detector_container_port = os.getenv("RT_DETR_PORT").strip('"')
    detector_url = f"http://rt-detr:{detector_container_port}"
    
    checkpoint = os.getenv("MVD_CHECKPOINT").strip('"')
    experiment_name=Path(os.environ.get("MVD_ARTIFACTS_DIR")).name
    checkpoint = str(Path("/models") / Path(experiment_name) / Path(checkpoint.lstrip("/")))
    
    STRIDE = 8  # 8x2=16 (16 stride, 32 context window)

    input_path = "/input"
    input_project_name = Path(os.environ.get('INPUT')).name
    project = VideoProject(input_path, mode=OpenMode.READ)

    output_path = "/output"
    output_predictions_path = Path(output_path) / Path(input_project_name) / Path("predictions")
    output_project_path = Path(output_path) / Path(input_project_name) / Path(f"predictions_SLY_format/{input_project_name}")
    output_project = VideoProject(str(output_project_path), mode=OpenMode.CREATE)
    output_meta = create_meta(class_names)
    output_project.set_meta(output_meta)

    checkpoint = Path(checkpoint)
    assert checkpoint.exists(), f"Checkpoint {checkpoint} does not exist."
    
    # Load models
    model, opts = load_mvd(checkpoint)
    detector = load_detector(session_url=detector_url)

    for dataset in project.datasets:
        dataset: VideoDataset
        output_dataset: VideoDataset = output_project.create_dataset(dataset.name, ds_path=dataset.path)
        for video_name, video_path, ann_path in dataset.items():
            video_results_dir = output_predictions_path / Path(dataset.path)
            # Predict
            predictions_raw = predict_video_with_detector(
                video_path,
                model,
                detector,
                opts,
                stride=STRIDE
            )
            predictions = postprocess_predictions(predictions_raw)

            # Visualize predictions
            import decord  # WARNING: if import decord in top, it will crash with 'Segmentation fault (core dumped)'
            vr = decord.VideoReader(video_path)
            fps = vr.get_avg_fps()
            frames_count = len(vr)
            frame_size = (vr[0].shape[0], vr[0].shape[1]) # h, w
            try:
                output_dataset.add_item_file(video_name, None, ann_from_predictions(frame_size, frames_count, predictions, output_meta, class_names))
            except Exception:
                print("Unable to save annotation in Supervisely format")
                traceback.print_exc()

            # Save predictions to JSON file
            os.makedirs(video_results_dir, exist_ok=True)
            output_json_path = video_results_dir / Path(f"predictions_{video_name}.json")
            with open(output_json_path, 'w') as f:
                json.dump(predictions, f, indent=4)

            draw_timeline(predictions_raw, fps, experiment_name=video_name, class_names=class_names, figsize=(15, 7), output_dir=video_results_dir)
            try:
                draw_class_segments_timeline(predictions_raw, fps, experiment_name=video_name, class_names=class_names, figsize=(15, 7), output_dir=video_results_dir)
            except Exception as e:
                traceback.print_exc()
    
            # Display additional statistics
            # Extract probabilities from the predictions list
            all_probs = np.array([pred['probabilities'] for pred in predictions_raw])
            num_windows = len(predictions_raw)
    
            # Get the most probable class for each window
            dominant_classes = np.argmax(all_probs, axis=1)
            class_counts = np.bincount(dominant_classes)
            top_classes = np.argsort(-class_counts)[:5]
            print("\nDominant classes in the video:")
            for cls in top_classes:
                if class_counts[cls] > 0:
                    percentage = (class_counts[cls] / num_windows) * 100
                    print(f"Class {cls}: {percentage:.2f}% ({class_counts[cls]} windows)")

            write_positive_fragments(predictions_raw, video_path, crop=True, output_dir=video_results_dir)
