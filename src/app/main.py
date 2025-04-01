import json
import os
from pathlib import Path
import time
import traceback
import dotenv

import supervisely as sly
import supervisely.io.env as env
from supervisely import ProjectMeta, VideoProject, OpenMode, VideoDataset, VideoAnnotation, VideoTagCollection
from supervisely.api.neural_network.model_api import ModelApi
from supervisely.project.download import download_async
from supervisely.io.fs import ensure_base_path
from tqdm import tqdm

from src.inference.inference import predict_video_with_detector, load_mvd, load_detector, postprocess_predictions


RT_DETR_SESSION_NAME = "rtdetr-mouse-detector"
RT_DETR_MODEL_DIR = "/experiments/835_MP: Images Sample for Detection Task (RTDETR2 - cat) Filtered and Splitted/1089_RT-DETRv2/"
REMOTE_MVD_MODEL_DIR = "/mouse-project/mvd-action-recognition/"
MVD_MODEL_DIR = "/models"
MVD_CHECKPOINT = "/models/MP_TRAIN_3_maximal_crop_2025-03-11_15-09-26/checkpoint-best/mp_rank_00_model_states.pt"
STRIDE = 8  # 8x2=16 (16 stride, 32 context window)
MODEL_CLASSES = ["idle", "Self-Grooming", "Head/Body TWITCH"]

dotenv.load_dotenv(os.path.expanduser("~/supervisely.env"))
dotenv.load_dotenv("local.env")


api = sly.Api()


def create_meta(class_names) -> ProjectMeta:
    tag_metas = [sly.TagMeta(class_name+"_prediction", sly.TagValueType.ANY_NUMBER) for class_name in class_names]
    meta = ProjectMeta(tag_metas=tag_metas)
    return meta


def merge_anns(source_ann: VideoAnnotation, new_ann: VideoAnnotation) -> VideoAnnotation:
    # only merge if there are no predictions in the source annotation
    if any([tag for tag in source_ann.tags if tag.meta.name.endswith("_prediction")]):
        return None
    source_ann = source_ann.clone(tags=source_ann.tags.add_items([tag for tag in new_ann.tags]))
    return source_ann

def ann_from_predictions(frame_size, frames_count, predictions, project_meta: ProjectMeta, class_names):
    print("frame size:", frame_size)
    label_to_tag_name = {i: class_name for i, class_name in enumerate(MODEL_CLASSES)}
    tags = []
    for prediction in predictions:
        tag_name = label_to_tag_name[prediction["label"]]
        pred_tag_name = tag_name + "_prediction"
        tag_meta = project_meta.get_tag_meta(pred_tag_name)
        if tag_meta is None:
            continue
        confidence = prediction["confidence"]
        frame_range = prediction["frame_range"]
        tag = sly.VideoTag(tag_meta, frame_range=frame_range, value=confidence)
        tags.append(tag)
    ann = VideoAnnotation(img_size=frame_size, frames_count=frames_count, tags=sly.VideoTagCollection(tags))
    return ann


def save_predictions(predictions, video_name):
    path = f"output/{video_name}.json"
    ensure_base_path(path)
    with open(path, "w") as f:
        json.dump(predictions, f, indent=4)
    api.file.upload(team_id=env.team_id(), src=path, dst=f"/mouse-predictions/{video_name}.json")

def inference_video(video_path, source_ann: VideoAnnotation, output_dataset: VideoDataset, output_meta, class_names, model, opts, detector, video_name=None, pbar=None):
    if any([tag for tag in source_ann.tags if tag.meta.name.endswith("_prediction")]):
        sly.logger.info(f"Skipping video {video_path} because it already has predictions")
        if pbar is not None:
            pbar.update(source_ann.frames_count)
        return False

    if video_name is None:
        video_name = Path(video_path).name
    # Predict
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
    save_predictions(predictions, video_name)

    # Visualize predictions
    import decord  # WARNING: if import decord in top, it will crash with 'Segmentation fault (core dumped)'
    vr = decord.VideoReader(video_path)
    frames_count = len(vr)
    frame_size = (vr[0].shape[0], vr[0].shape[1]) # h, w
    try:
        annotation = merge_anns(source_ann, ann_from_predictions(frame_size, frames_count, predictions, output_meta, class_names))
        if annotation is not None:
            output_dataset.add_item_file(video_name, None, annotation)
        return True
    except Exception:
        sly.logger.warning("Unable to save annotation in Supervisely format", exc_info=True)
        traceback.print_exc()
        return False


def get_total(project: VideoProject):
    import decord
    total = 0
    for dataset in project.datasets:
        for _, video_path, _ in dataset.items():
            vr = decord.VideoReader(video_path)
            frames_count = len(vr)
            total += frames_count
    return total

def inference_project(project: VideoProject, project_name: str, model, opts, detector):
    class_names = ["Self-Grooming", "Head/Body TWITCH"]

    model_meta = create_meta(class_names)
    project_meta = project.meta.merge(model_meta)
    project.set_meta(project_meta)

    updated_anns = []
    total = get_total(project)
    with tqdm(total=total, desc=f'Predicting on project "{project_name}"') as pbar:
        for dataset in project.datasets:
            dataset: VideoDataset
            for _, video_path, ann_path in dataset.items():
                source_ann = VideoAnnotation.load_json_file(ann_path, project_meta)
                if inference_video(video_path, source_ann, dataset, project_meta, class_names, model, opts, detector, pbar=pbar):
                    updated_anns.append(ann_path)
    return updated_anns

def get_or_create_session(api: sly.Api) -> ModelApi:
    rt_detr_slug = "supervisely-ecosystem/RT-DETRv2/supervisely_integration/serve"
    team_id = env.team_id()
    apps = api.app.get_list(team_id=team_id, only_running=True)
    for app in apps:
        if rt_detr_slug.lower() == app.slug.lower():
            for task in app.tasks:
                print(json.dumps(task, indent=4))
                if task["meta"]["name"] == RT_DETR_SESSION_NAME:
                    return api.nn.connect(task["id"])
    agents = api.agent.get_list_available(team_id, has_gpu=True)
    if len(agents) == 0:
        raise RuntimeError("No agents with GPU available")
    agent = agents[0]
    module_id = api.app.get_ecosystem_module_id(rt_detr_slug.lower())
    task_info = api.task.start(agent_id=agent.id, workspace_id=env.workspace_id(), module_id=module_id, task_name=RT_DETR_SESSION_NAME, app_version="deploy-api-test", is_branch=True, log_level="debug")
    sly.logger.info("Sleepting for 2 minutes to wait for the session to start")
    time.sleep(60*2)
    api.task.wait(id=task_info["id"], target_status=api.task.Status.STARTED, wait_attempts=100, wait_attempt_timeout_sec=5)
    api.nn.deploy.load_custom_model(task_info["id"], team_id=team_id, artifacts_dir=RT_DETR_MODEL_DIR, checkpoint_name="best.pth", runtime="PyTorch", device="cuda")
    model = api.nn.connect(task_info["id"])
    return model

def check_and_update_ann(ann_path, project_meta):
    video_annotation = VideoAnnotation.from_json(json.load(open(ann_path, 'r')), project_meta)
    has_predictions = False
    filtered_tags = []
    for tag in video_annotation.tags:
        if tag.meta.name == "confidence":
            filtered_tags.append(tag)
        if tag.meta.name.endswith("_prediction"):
            has_predictions = True
            filtered_tags.append(tag)
    if has_predictions:
        video_annotation = video_annotation.clone(tags=VideoTagCollection(filtered_tags))
        with open(ann_path, "w") as f:
            json.dump(video_annotation.to_json(), f, indent=4)
    return has_predictions


def main():
    team_id = env.team_id()
    project_id = env.project_id()

    # Load models
    sly.logger.info("Creating session with detector")
    detector = get_or_create_session(api)
    sly.logger.info("Loading model")
    size = api.file.get_directory_size(team_id=team_id, path=REMOTE_MVD_MODEL_DIR)
    with tqdm(total=size, desc="Downloading MVD model", unit="B", unit_scale=True, unit_divisor=1024) as pbar:
        api.file.download_directory(team_id=team_id, remote_path=REMOTE_MVD_MODEL_DIR, local_save_path=MVD_MODEL_DIR, progress_cb=pbar.update)
    model, opts = load_mvd(MVD_CHECKPOINT)

    # Download project
    project_path = "input/project"
    sly.logger.info(f"Downloading project {project_id}")
    download_async(api, project_id, dest_dir=project_path, save_video_info=True)

    # Inference
    project_info = api.project.get_info_by_id(project_id)
    project = VideoProject(project_path, mode=OpenMode.READ)
    updated_ann_paths = inference_project(project, project_name=project_info.name, model=model, opts=opts, detector=detector)

    sly.logger.info("Uploading annotations")
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
                if ann_path in updated_ann_paths and check_and_update_ann(ann_path, project.meta):
                    video_info = dataset.get_item_info(item_name=video_name)
                    video_ids.append(video_info.id)
                    ann_paths.append(ann_path)
        api.video.annotation.upload_paths(video_ids=video_ids, ann_paths=ann_paths, project_meta=project.meta, progress_cb=pbar.update)

if __name__ == "__main__":
    main()
