import json
import os
from pathlib import Path
import time
import traceback
import dotenv

import supervisely as sly
from supervisely.nn.inference.session import Session
import supervisely.io.env as env
from supervisely import ProjectMeta, VideoProject, OpenMode, VideoDataset, VideoAnnotation
from supervisely.project.download import download_async
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


def create_meta(class_names) -> ProjectMeta:
    tag_metas = [sly.TagMeta(class_name+"_prediction", sly.TagValueType.NONE) for class_name in class_names] + [sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER)]
    meta = ProjectMeta(tag_metas=tag_metas)
    return meta


def merge_anns(source_ann: VideoAnnotation, new_ann: VideoAnnotation) -> VideoAnnotation:
    new_ann = new_ann.clone(tags=new_ann.tags.add_items([tag for tag in source_ann.tags]))
    return new_ann

def ann_from_predictions(frame_size, frames_count, predictions, project_meta: ProjectMeta, class_names):
    print("frame size:", frame_size)
    label_to_tag_meta = {i: project_meta.get_tag_meta(class_name+"_prediction") for i, class_name in enumerate(class_names)}
    conf_tag_meta = project_meta.get_tag_meta("confidence")
    tags = []
    for prediction in predictions:
        tag_meta = label_to_tag_meta[prediction["label"]]
        if tag_meta is None:
            continue
        confidence = prediction["confidence"]
        frame_range = prediction["frame_range"]
        tag = sly.VideoTag(tag_meta, frame_range=frame_range)
        conf_tag = sly.VideoTag(conf_tag_meta, value=confidence, frame_range=frame_range)
        tags.extend([tag, conf_tag])
    ann = VideoAnnotation(img_size=frame_size, frames_count=frames_count, tags=sly.VideoTagCollection(tags))
    return ann

def inference_video(video_path, source_ann: VideoAnnotation, output_dataset: VideoDataset, output_meta, class_names, model, opts, detector, video_name=None, pbar=None):
    if any([tag for tag in source_ann.tags if tag.meta.name.endswith("_prediction")]):
        sly.logger.info(f"Skipping video {video_path} because it already has predictions")
        if pbar is not None:
            pbar.update(source_ann.frames_count)
        return
    
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

    # Visualize predictions
    import decord  # WARNING: if import decord in top, it will crash with 'Segmentation fault (core dumped)'
    vr = decord.VideoReader(video_path)
    frames_count = len(vr)
    frame_size = (vr[0].shape[0], vr[0].shape[1]) # h, w
    try:
        output_dataset.add_item_file(video_name, None, merge_anns(source_ann, ann_from_predictions(frame_size, frames_count, predictions, output_meta, class_names)))
    except Exception:
        sly.logger.warning("Unable to save annotation in Supervisely format", exc_info=True)
        traceback.print_exc()


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

    total = get_total(project)
    with tqdm(total=total, desc="Inference") as pbar:
        for dataset in project.datasets:
            dataset: VideoDataset
            for _, video_path, ann_path in dataset.items():
                full_video_path = Path(project.parent_dir) / Path(project.name) / Path(dataset.path) / Path(video_path)
                source_ann = VideoAnnotation.load_json_file(ann_path, project_meta)
                inference_video(str(full_video_path), source_ann, dataset, project_meta, class_names, model, opts, detector, pbar=pbar)

def get_or_create_session(api: sly.Api) -> Session:
    rt_detr_slug = "supervisely-ecosystem/RT-DETRv2/supervisely_integration/serve"
    team_id = env.team_id()
    apps = api.app.get_list(team_id=team_id, only_running=True)
    for app in apps:
        if rt_detr_slug.lower() == app.slug.lower():
            for task in app.tasks:
                print(json.dumps(task, indent=4))
                if task["meta"]["name"] == RT_DETR_SESSION_NAME:
                    return Session(api, task["id"])
    agents = api.agent.get_list_available(team_id, has_gpu=True)
    if len(agents) == 0:
        raise RuntimeError("No agents with GPU available")
    agent = agents[0]
    module_id = api.app.get_ecosystem_module_id(rt_detr_slug.lower())
    task_info = api.task.start(agent_id=agent.id, workspace_id=env.workspace_id(), module_id=module_id, task_name=RT_DETR_SESSION_NAME)
    time.sleep(60*5)
    api.task.wait(id=task_info["id"], target_status=api.task.Status.STARTED, wait_attempts=100, wait_attempt_timeout_sec=5)
    api.nn.deploy.load_custom_model(task_info["id"], team_id=team_id, artifacts_dir=RT_DETR_MODEL_DIR, checkpoint_name="best.pth", runtime="PyTorch", device="cuda")
    session = Session(api, task_info["id"])
    return session

def main():
    api = sly.Api()
    team_id = env.team_id()
    project_id = env.project_id()

    # Load models
    api.file.download_directory(team_id=team_id, remote_path=REMOTE_MVD_MODEL_DIR, local_save_path=MVD_MODEL_DIR)
    session = get_or_create_session(api)
    detector = load_detector(session_url=session.base_url)
    model, opts = load_mvd(MVD_CHECKPOINT)

    # Download project
    project_path = "input/project"
    download_async(api, project_id, dest_dir=project_path, save_video_info=True)

    # Inference
    project_info = api.project.get_info_by_id(project_id)
    project = VideoProject(project_path, mode=OpenMode.READ)
    inference_project(project, project_name=project_info.name, model=model, opts=opts, detector=detector)

    # Upload results
    for dataset in project.datasets:
        dataset: VideoDataset
        video_ids = []
        ann_paths = []
        for video_name, video_path, ann_path in dataset.items():
            video_info = dataset.get_item_info(item_name=video_name)
            video_ids.append(video_info.id)
            ann_paths.append(ann_path)

        api.video.annotation.upload_paths(video_ids=video_ids, ann_paths=ann_paths, project_meta=project.meta)


if __name__ == "__main__":
    main()
