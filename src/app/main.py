import json
import os
from pathlib import Path
import time
import traceback
import dotenv

import supervisely as sly
import supervisely.io.env as env
from supervisely import ProjectMeta, VideoProject, OpenMode, VideoDataset, VideoAnnotation, VideoTagCollection
from supervisely.nn.model.model_api import ModelAPI
from supervisely.project.download import download_async
from supervisely.io.fs import ensure_base_path, mkdir, silent_remove
from supervisely.io.json import load_json_file
from tqdm import tqdm

from src.inference.inference import predict_video_with_detector, load_mvd, postprocess_predictions


RT_DETR_SESSION_NAME = "rtdetr-mouse-detector"
REMOTE_RT_DETR_CHECKPOINT_PATH = os.getenv("modal.state.detectorCheckpointPath")
REMOTE_RT_DETR_MODEL_DIR = str(Path(REMOTE_RT_DETR_CHECKPOINT_PATH).parent.parent)
RT_DETR_CHECKPOINT_NAME = str(Path(REMOTE_RT_DETR_CHECKPOINT_PATH).name)
MVD_MODEL_DIR = "/models/mvd-action-recognition"
REMOTE_MVD_CHECKPOINT_PATH = os.getenv("modal.state.MVDCheckpointPath")
REMOTE_MVD_MODEL_DIR = str(Path(REMOTE_MVD_CHECKPOINT_PATH).parent.parent)
MVD_CHECKPOINT_NAME = str(Path(REMOTE_MVD_CHECKPOINT_PATH).name)
MVD_CHECKPOINT = REMOTE_MVD_CHECKPOINT_PATH.replace(REMOTE_MVD_MODEL_DIR, MVD_MODEL_DIR)
STRIDE = 8  # 8x2=16 (16 stride, 32 context window)
MODEL_CLASSES = ["idle", "Head/Body TWITCH", "Self-Grooming"]  # classes have been swapped to match those in training
RT_DETR_STOP_SESSION_FLAG = False

dotenv.load_dotenv(os.path.expanduser("~/supervisely.env"))
dotenv.load_dotenv("local.env")


api = sly.Api()


def create_model_meta(class_names) -> ProjectMeta:
    tag_metas = [sly.TagMeta(class_name, sly.TagValueType.ANY_NUMBER) for class_name in class_names]
    meta = ProjectMeta(tag_metas=tag_metas)
    return meta

def merge_anns(source_ann: VideoAnnotation, new_ann: VideoAnnotation) -> VideoAnnotation:
    """Do not use, as video annotations can only be appended to existing ones on instance. Merging is not needed"""
    src_tags = [tag for tag in source_ann.tags]
    new_tags = [tag for tag in new_ann.tags]
    merged_tags = src_tags + new_tags
    merged_tags_col = sly.VideoTagCollection(merged_tags)
    source_ann = source_ann.clone(tags=merged_tags_col)
    return source_ann

def ann_from_predictions(frame_size, frames_count, predictions, project_meta: ProjectMeta):
    print("frame size:", frame_size)
    label_to_tag_name = {i: class_name for i, class_name in enumerate(MODEL_CLASSES)}
    tags = []
    for prediction in predictions:
        tag_name = label_to_tag_name[prediction["label"]]
        pred_tag_name = tag_name
        tag_meta = project_meta.get_tag_meta(pred_tag_name)
        assert tag_meta is not None, f"Tag meta '{pred_tag_name}' not found in project meta"
        confidence = prediction["confidence"]
        frame_range = prediction["frame_range"]
        tag = sly.VideoTag(tag_meta, frame_range=frame_range, value=confidence)
        tags.append(tag)
    ann = VideoAnnotation(img_size=frame_size, frames_count=frames_count, tags=sly.VideoTagCollection(tags))
    return ann

def read_or_redownload_video(dataset: VideoDataset, video_name: str, video_path: str):
    import decord

    def try_open(path: str):
        try:
            vr = decord.VideoReader(path)
            return len(vr)
        except Exception:
            return None

    frames_count = try_open(video_path)
    if frames_count is not None:
        return frames_count

    sly.logger.warning(f"Video '{video_name}' is unreadable. Trying to re-download...")
    info = dataset.get_item_info(item_name=video_name)

    try:
        silent_remove(video_path)
        api.video.download_path(info.id, video_path)
    except Exception as e:
        sly.logger.warning(f"Re-download attempt failed for '{video_name}': {e}")
        return None
    
    frames_count = try_open(video_path)
    if frames_count is not None:
        sly.logger.info(f"Successfully re-downloaded and read video '{video_name}'")
        return frames_count

    sly.logger.warning(f"File '{video_name}' still unreadable after re-download attempt")
    return None

# def save_predictions(predictions, video_name):
#     path = f"output/{video_name}.json"
#     ensure_base_path(path)
#     with open(path, "w") as f:
#         json.dump(predictions, f, indent=4)
#     api.file.upload(team_id=env.team_id(), src=path, dst=f"/mouse-predictions/{video_name}.json")

def inference_video(video_path, source_ann: VideoAnnotation, output_dataset: VideoDataset, output_meta, model, opts, detector: ModelAPI, video_name=None, pbar=None):
    if video_name is None:
        video_name = Path(video_path).name

    # Predict
    try:
        predictions_raw = predict_video_with_detector(
            video_path,
            model,
            detector,
            opts,
            stride=STRIDE,
            pbar=pbar
        )
    except Exception as e:
        if detector.is_deployed():
            raise
        detector = get_or_create_session(api)
        predictions_raw = predict_video_with_detector(
            video_path,
            model,
            detector,
            opts,
            stride=STRIDE,
            pbar=pbar
        )
    predictions = postprocess_predictions(predictions_raw)

    # save predictions to teamfiles:
    # save_predictions(predictions, video_name)

    import decord  # WARNING: if import decord in top, it will crash with 'Segmentation fault (core dumped)'
    vr = decord.VideoReader(video_path)
    frames_count = len(vr)
    frame_size = (vr[0].shape[0], vr[0].shape[1]) # h, w
    annotation = ann_from_predictions(frame_size, frames_count, predictions, output_meta)
    # annotation = merge_anns(source_ann, new_ann)
    output_dataset.add_item_file(video_name, None, annotation)
    return annotation


def get_total(project: VideoProject):
    total = 0

    for dataset in project.datasets:
        dataset: VideoDataset
        with tqdm(total=len(dataset.items()), desc=f"Getting total frames for dataset: '{dataset.name}'") as pbar:
            for video_name, video_path, _ in dataset.items():
                frames_count = read_or_redownload_video(dataset, video_name, video_path)
                if frames_count is None:
                    pbar.update(1)
                    continue
                total += frames_count
                pbar.update(1)
    return total

def inference_project(project: VideoProject, project_name: str, model, opts, detector):
    class_names = ["Self-Grooming", "Head/Body TWITCH"]

    model_meta = create_model_meta(class_names)
    project_meta = project.meta.merge(model_meta)
    project.set_meta(project_meta)

    updated_anns = []
    total = get_total(project)
    with tqdm(total=total, desc=f'Predicting on project "{project_name}"') as pbar:
        for dataset in project.datasets:
            dataset: VideoDataset
            for video_name, video_path, ann_path in dataset.items():
                source_ann = VideoAnnotation.load_json_file(ann_path, project_meta)
                try:
                    inference_video(video_path, source_ann, dataset, project_meta, model, opts, detector, pbar=pbar)
                    updated_anns.append(ann_path)
                except Exception as e:
                    sly.logger.warning(f"Skipping inference for '{video_name}' due to read error: {e}")
                    continue
    return updated_anns

def get_or_create_session(api: sly.Api) -> ModelAPI:
    global RT_DETR_STOP_SESSION_FLAG
    rt_detr_slug = "supervisely-ecosystem/RT-DETRv2/supervisely_integration/serve"
    team_id = env.team_id()
    apps = api.app.get_list(team_id=team_id, only_running=True)
    for app in apps:
        if rt_detr_slug.lower() == app.slug.lower():
            for task in app.tasks:
                if task["meta"]["name"] == RT_DETR_SESSION_NAME:
                    RT_DETR_STOP_SESSION_FLAG = False
                    model = api.nn.connect(task["id"])
                    if not model.is_deployed():
                        model.load(REMOTE_RT_DETR_CHECKPOINT_PATH, runtime="PyTorch", device="cuda")
                    return model
    agents = api.agent.get_list_available(team_id, has_gpu=True)
    if len(agents) == 0:
        raise RuntimeError("No agents with GPU available")
    agent = agents[0]
    model = api.nn.deploy(model=REMOTE_RT_DETR_CHECKPOINT_PATH, runtime="PyTorch", device="cuda", agent_id=agent.id, workspace_id=env.workspace_id(), task_name=RT_DETR_SESSION_NAME)
    RT_DETR_STOP_SESSION_FLAG = True
    return model

def check_and_update_ann(ann_path, project_meta):
    src_ann_json = load_json_file(ann_path)
    video_annotation = VideoAnnotation.from_json(src_ann_json, project_meta)
    has_predictions = False
    filtered_tags = []
    for tag in video_annotation.tags:
        if tag.meta.name == "confidence":
            filtered_tags.append(tag)
            has_predictions = True
        # if tag.meta.name.endswith("_prediction"):
        #     has_predictions = True
        #     filtered_tags.append(tag)
    if has_predictions:
        video_annotation = video_annotation.clone(tags=VideoTagCollection(filtered_tags))
        with open(ann_path, "w") as f:
            json.dump(video_annotation.to_json(), f, indent=4)
    return has_predictions


def main():
    team_id = env.team_id()
    project_id = env.project_id()
    dataset_id = env.dataset_id(raise_not_found=False)
    dataset_ids = [dataset_id] if dataset_id is not None else None

    # Load models
    sly.logger.info("Creating session with detector")
    detector = get_or_create_session(api)
    sly.logger.info("Loading model")
    mvd_checkpoint_file_path = REMOTE_MVD_CHECKPOINT_PATH
    mvd_checkpoint_file_fino = api.file.get_info_by_path(team_id=team_id, remote_path=mvd_checkpoint_file_path)
    mvd_config_file_path = Path(REMOTE_MVD_MODEL_DIR, "config.txt").as_posix()
    mvd_config_file_info = api.file.get_info_by_path(team_id=team_id, remote_path=mvd_config_file_path)
    mvd_config_local_path = str(Path(MVD_MODEL_DIR, "config.txt"))
    size = mvd_checkpoint_file_fino.sizeb + mvd_config_file_info.sizeb
    with tqdm(total=size, desc="Downloading MVD model", unit="B", unit_scale=True, unit_divisor=1024) as pbar:
        api.file.download(team_id=team_id, remote_path=mvd_checkpoint_file_path, local_save_path=MVD_CHECKPOINT, progress_cb=pbar.update)
        api.file.download(team_id=team_id, remote_path=mvd_config_file_path, local_save_path=mvd_config_local_path, progress_cb=pbar.update)
    
    # Swap classes back if this is the original MVD model (trained with the old class order)
    experiment_info_path = Path(REMOTE_MVD_MODEL_DIR, "experiment_info.json").as_posix()
    if not api.file.exists(team_id=team_id, remote_path=experiment_info_path):
        global MODEL_CLASSES
        MODEL_CLASSES = ["idle", "Self-Grooming", "Head/Body TWITCH"]
        sly.logger.info("Detected the original MVD model, changing class order")
    model, opts = load_mvd(MVD_CHECKPOINT)

    # Download project
    project_path = "input/project"
    if os.path.exists(project_path):
        mkdir(project_path, True)

    sly.logger.info(f"Downloading project {project_id}{'' if dataset_id is None else ', dataset ' + str(dataset_id)}")
    download_async(api, project_id, dest_dir=project_path, dataset_ids=dataset_ids, save_video_info=True)

    # Inference
    project_info = api.project.get_info_by_id(project_id)
    project = VideoProject(project_path, mode=OpenMode.READ)
    updated_ann_paths = inference_project(project, project_name=project_info.name, model=model, opts=opts, detector=detector)

    if len(updated_ann_paths) == 0:
        sly.logger.info("No annotations were updated")
        return

    # Upload results
    api.project.update_meta(project_id, project.meta)
    with tqdm(total=len(updated_ann_paths), desc="Uploading annotations") as pbar:
        video_ids = []
        ann_paths = []
        for dataset in project.datasets:
            dataset: VideoDataset
            for video_name, _, ann_path in dataset.items():
                if ann_path in updated_ann_paths:
                    video_info = dataset.get_item_info(item_name=video_name)
                    video_ids.append(video_info.id)
                    ann_paths.append(ann_path)
        api.video.annotation.upload_paths(video_ids=video_ids, ann_paths=ann_paths, project_meta=project.meta, progress_cb=pbar.update)

    if RT_DETR_STOP_SESSION_FLAG:
        sly.logger.info("Stopping RT-DETR session")
        detector.shutdown()

if __name__ == "__main__":
    main()
