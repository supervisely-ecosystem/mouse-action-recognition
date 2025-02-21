import supervisely as sly

dataset_id = 2544

api = sly.Api()
workspace_id = 21
team_id = api.workspace.get_info_by_id(workspace_id).team_id

project_id = api.dataset.get_info_by_id(dataset_id).project_id
project_info = api.project.get_info_by_id(project_id)
project_meta_json = api.project.get_meta(project_id)
project_meta = sly.ProjectMeta.from_json(project_meta_json)

# download all video from dataset
sly.download_video_project(api, project_id, "HOM Mice F.2632_HOM 12 Days post tre", [dataset_id])