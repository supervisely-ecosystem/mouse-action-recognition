import os
import json
import traceback
from pathlib import Path
from typing import List
from supervisely import (
    VideoProject,
    OpenMode,
    logger,
    VideoDataset,
    VideoAnnotation,
    ProjectMeta,
    TagMeta,
    TagValueType,
    VideoTag,
    VideoTagCollection,
)

from inference_script.run_inference import (
    create_meta,
    inference_directory,
    inference_video,
)
from src.inference.inference import postprocess_predictions
from supervisely_integration.train.scripts.load import load_mvd
from supervisely_integration.train.scripts.predictor import predict_video
from supervisely.io.json import dump_json_file

STRIDE = 8


def run_inference(input_dir: str, output_dir: str, model_meta: ProjectMeta, checkpoint_path: str, config_path: str):
    tag_names = ["idle", "Self-Grooming", "Head/Body TWITCH"]
    logger.info("Loading model")
    model, opts = load_mvd(checkpoint_path, config_path)
    logger.info("Predicting test videos")
    inference_directory(input_dir, output_dir, tag_names, model, opts, model_meta)

def inference_directory(
    input_dir: str, output_dir: str, tag_names: List[str], model, opts, model_meta: ProjectMeta
):
    output_meta = create_meta(tag_names)
    for video_name in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_name)
        if os.path.isfile(video_path):
            inference_video(
                video_path,
                video_name,
                output_dir,
                output_meta,
                tag_names,
                model,
                opts,
                model_meta,
            )


def inference_video(
    video_path,
    video_name,
    output_dir,
    output_meta,
    tag_names,
    model,
    opts,
    model_meta,
):
    video_ann_path = os.path.join(output_dir, f"{video_name}.json")
    os.makedirs(str(output_dir), exist_ok=True)

    # Predict
    predictions_raw = predict_video(video_path, model, opts, model_meta, stride=STRIDE)
    predictions = postprocess_predictions(predictions_raw)

    # ------ REMOVE? ------- #
    # Visualize predictions
    import decord
    vr = decord.VideoReader(video_path)
    frames_count = len(vr)
    frame_size = (vr[0].shape[0], vr[0].shape[1])  # h, w
    try:
        ann_from_predictions(frame_size, frames_count, predictions, output_meta, tag_names)
    except Exception:
        print("Unable to save annotation in Supervisely format")
        traceback.print_exc()
    # ---------------------- #

    # Save predictions to JSON file
    dump_json_file(predictions, video_ann_path)
    print(f"Saved predictions to {video_ann_path}")


def create_meta(tag_names) -> ProjectMeta:
    tag_metas = [TagMeta(class_name, TagValueType.NONE) for class_name in tag_names] + [
        TagMeta("confidence", TagValueType.ANY_NUMBER)
    ]
    meta = ProjectMeta(tag_metas=tag_metas)
    return meta


def ann_from_predictions(
    frame_size, frames_count, predictions, project_meta: ProjectMeta, tag_names
):
    print("frame size:", frame_size)
    label_to_tag_meta = {
        i: project_meta.get_tag_meta(class_name)
        for i, class_name in enumerate(tag_names)
    }
    conf_tag_meta = project_meta.get_tag_meta("confidence")
    tags = []
    for prediction in predictions:
        tag_meta = label_to_tag_meta[prediction["label"]]
        confidence = prediction["confidence"]
        frame_range = prediction["frame_range"]
        tag = VideoTag(tag_meta, frame_range=frame_range)
        conf_tag = VideoTag(conf_tag_meta, value=confidence, frame_range=frame_range)
        tags.extend([tag, conf_tag])
    ann = VideoAnnotation(
        img_size=frame_size, frames_count=frames_count, tags=VideoTagCollection(tags)
    )
    return ann
