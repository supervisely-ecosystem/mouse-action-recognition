import os
from typing import List
from src.inference.inference import postprocess_predictions
from supervisely import (
    ProjectMeta,
    TagMeta,
    TagValueType,
    logger,
)
from supervisely.app.widgets import Progress
from supervisely.io.json import dump_json_file
from supervisely_integration.train.scripts.loader import load_mvd
from supervisely_integration.train.scripts.predictor import predict_video

STRIDE = 8


def run_inference(
    input_dir: str,
    output_dir: str,
    model_meta: ProjectMeta,
    checkpoint_path: str,
    config_path: str,
    progress_bar_main: Progress,
    progress_bar_secondary: Progress,
):
    tag_names = ["idle", "Self-Grooming", "Head/Body TWITCH"]
    logger.info("Loading model")
    model, opts = load_mvd(checkpoint_path, config_path)
    logger.info("Predicting test videos")
    inference_directory(
        input_dir,
        output_dir,
        tag_names,
        model,
        opts,
        model_meta,
        progress_bar_main,
        progress_bar_secondary,
    )


def inference_directory(
    input_dir: str,
    output_dir: str,
    tag_names: List[str],
    model,
    opts,
    model_meta: ProjectMeta,
    progress_bar_main: Progress,
    progress_bar_secondary: Progress,
):
    output_meta = create_meta(tag_names)
    all_videos = os.listdir(input_dir)
    with progress_bar_main(
        message="Inference test videos with MVD model", total=len(all_videos)
    ) as pbar:
        progress_bar_main.show()
        for video_name in all_videos:
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
                    progress_bar_secondary,
                )
            pbar.update(1)
        progress_bar_main.hide()


def inference_video(
    video_path,
    video_name,
    output_dir,
    output_meta,
    tag_names,
    model,
    opts,
    model_meta,
    progress_bar_secondary,
):
    video_ann_path = os.path.join(output_dir, f"{video_name}.json")
    os.makedirs(str(output_dir), exist_ok=True)

    # Predict
    predictions_raw = predict_video(
        video_path,
        model,
        opts,
        model_meta,
        stride=STRIDE,
        progress_bar=progress_bar_secondary,
    )
    predictions = postprocess_predictions(predictions_raw)

    # Save predictions to JSON file
    dump_json_file(predictions, video_ann_path)
    print(f"Saved predictions to {video_ann_path}")


def create_meta(tag_names) -> ProjectMeta:
    tag_metas = [TagMeta(class_name, TagValueType.NONE) for class_name in tag_names] + [
        TagMeta("confidence", TagValueType.ANY_NUMBER)
    ]
    meta = ProjectMeta(tag_metas=tag_metas)
    return meta
