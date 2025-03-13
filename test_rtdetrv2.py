import os

# from dotenv import load_dotenv

import supervisely as sly

# if sly.is_development():
#     load_dotenv("local.env")
#     load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

# Connect to the serving session
session = sly.nn.inference.Session(api, session_url="http://supervisely-utils-rtdetrv2-inference-1:8000")

# ann = session.inference_image_path("results/timeline_visualization.png")
print(session.get_deploy_info())