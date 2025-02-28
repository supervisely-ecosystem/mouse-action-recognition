import supervisely as sly

# dataset_id = 2544
project_id = 944

api = sly.Api()
# workspace_id = 21
# team_id = api.workspace.get_info_by_id(workspace_id).team_id

# project_id = api.dataset.get_info_by_id(dataset_id).project_id
# project_info = api.project.get_info_by_id(project_id)
# project_meta_json = api.project.get_meta(project_id)
# project_meta = sly.ProjectMeta.from_json(project_meta_json)

# download all video from dataset
# MP_TRAIN
from supervisely.project.video_project import download_video_project
download_video_project(api, project_id=project_id, dest_dir="MP_TRAIN_3")

# video_ids = []
# ann_ids = []
# for parents, dataset in api.dataset.tree(project_id):
#     infos = api.video.get_list(dataset.id)
#     for v_info in infos:
#         v_name = v_info.name
#         v_id = v_info.id
#         video_ids.append(v_id)

#     video_ids += [info.id for info in infos]
#     # v_name = 
#     paths = []

# api.video.download_paths_async(video_ids, "MP_TRAIN")
# sly.download_video_project(api, project_id=project_id, dest_dir="MP_TRAIN", dataset_ids=ids)