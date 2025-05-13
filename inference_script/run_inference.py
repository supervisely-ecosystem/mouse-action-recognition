from pathlib import Path
import json
import os

import traceback
from typing import List
import supervisely as sly
from supervisely import VideoProject, OpenMode, VideoDataset, ProjectMeta, VideoAnnotation

from src.inference.inference import predict_video_with_detector, load_mvd, load_detector, postprocess_predictions


STRIDE = 8  # 8x2=16 (16 stride, 32 context window)


# os.environ["SERVER_ADDRESS"] = ""
# os.environ["API_TOKEN"] = ""
# os.environ["TEAM_ID"] = "1"


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

def inference_video(video_path, output_predictions_path, output_dataset: VideoDataset, output_meta, class_names, model, opts, detector, video_name=None):
    if video_name is None:
        video_name = Path(video_path).name
    video_prediction_path = output_predictions_path / Path(f"{video_name}.json")
    os.makedirs(str(output_predictions_path), exist_ok=True)
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
    frames_count = len(vr)
    frame_size = (vr[0].shape[0], vr[0].shape[1]) # h, w
    try:
        output_dataset.add_item_file(video_name, None, ann_from_predictions(frame_size, frames_count, predictions, output_meta, class_names))
    except Exception:
        print("Unable to save annotation in Supervisely format")
        traceback.print_exc()

    # Save predictions to JSON file
    with open(video_prediction_path, 'w') as f:
        json.dump(predictions, f, indent=4)
    print(f"Saved predictions to {video_prediction_path}")


def inference_project(project: VideoProject, project_name: str, output_dir: str, class_names: List[str], model, opts, detector):
    output_predictions_path = Path(output_dir) / Path(project_name) / Path("predictions")
    output_project_path = Path(output_dir) / Path(project_name) / Path("SLY_project")
    output_project = VideoProject(str(output_project_path), mode=OpenMode.CREATE)
    output_meta = create_meta(class_names)
    output_project.set_meta(output_meta)

    for dataset in project.datasets:
        dataset: VideoDataset
        output_dataset: VideoDataset = output_project.create_dataset(dataset.name, ds_path=dataset.path)
        ds_predictions_path = output_predictions_path / Path(dataset.path)
        for _, video_path, _ in dataset.items():
            full_video_path = Path(project.parent_dir) / Path(project.name) / Path(dataset.path) / Path(video_path)
            inference_video(str(full_video_path), ds_predictions_path, output_dataset, output_meta, class_names, model, opts, detector)


def inference_directory(input_directory: str, project_name: str, output_dir: str, class_names: List[str], model, opts, detector):
    output_predictions_path = Path(output_dir) / Path(project_name) / Path("predictions")
    output_project_path = Path(output_dir) / Path(project_name) / Path("SLY_project")
    output_project = VideoProject(str(output_project_path), mode=OpenMode.CREATE)
    output_meta = create_meta(class_names)
    output_project.set_meta(output_meta)
    output_dataset = output_project.create_dataset("predictions")

    for video_path in os.listdir(input_directory):
        full_video_path = str(Path(input_directory) / Path(video_path))
        if os.path.isfile(full_video_path):
            inference_video(full_video_path, output_predictions_path, output_dataset, output_meta, class_names, model, opts, detector)


if __name__ == '__main__':
    class_names = ["idle", "Self-Grooming", "Head/Body TWITCH"]

    detector_container_port = 8000
    detector_url = f"http://rt-detr:{detector_container_port}"
    
    model_dir = os.getenv("MODEL_DIR")
    checkpoint = f"{model_dir}/checkpoint-best/mp_rank_00_model_states.pt"
    input_path = "/input"
    output_dir = "/output"
    
    input_dir_name = Path(os.environ.get('INPUT')).name

    is_directory = False
    is_project = False
    project = None
    if os.path.isdir(input_path):
        is_directory = True
        try:
            project = VideoProject(input_path, mode=OpenMode.READ)
            is_project = True
        except:
            pass

    # Load models
    model, opts = load_mvd(checkpoint)
    detector = load_detector(session_url=detector_url)

    if is_project:
        print("Predicting project")
        inference_project(project, input_dir_name, output_dir, class_names, model, opts, detector)
    elif is_directory:
        print("Predicting directory")
        inference_directory(input_path, input_dir_name, output_dir, class_names, model, opts, detector)
    else:
        print("Predicting video")
        input_video_name = os.path.splitext(input_dir_name)[0]
        output_predictions_path = Path(output_dir) / Path(input_video_name) / Path("predictions")
        output_project_path = Path(output_dir) / Path(input_video_name) / Path("SLY_project")
        output_project = VideoProject(str(output_project_path), mode=OpenMode.CREATE)
        output_meta = create_meta(class_names)
        output_project.set_meta(output_meta)
        output_dataset = output_project.create_dataset("predictions")
        inference_video(input_path, output_predictions_path, output_dataset, output_meta, class_names, model, opts, detector, input_video_name)
