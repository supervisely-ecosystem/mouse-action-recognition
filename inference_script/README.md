# How to run inference

You can run inference in a Docker container on your local machine. We have built Docker images for the MVD model and the mouse detector, and prepared a docker-compose file. This way you can run the inference without manully installing any dependencies.

1. Clone our github repository and navigate to the `inference_script` directory:

```bash
git clone https://github.com/supervisely-ecosystem/mouse-action-recognition
cd mouse-action-recognition/inference_script
```

2. Download both of your models: the trained MVD model and the mouse detector. Place the models in the `inference_script/models` directory.

3. Open `inference_script/.env` file and define your paths to an input directory with videos and a directory to the trained MVD model (the last contains all files needed to load the model, such as config.txt and a folder with best checkpoint). Here is an example of the `.env` file:

```bash
# Note: all paths are relative to this .env file (you can use absolute paths as well)
# Directory where the MVD model is stored
MODEL_DIR=./models/MP_TRAIN_3_maximal_crop_2025-03-11_15-09-26
# Path to an input directory with videos
INPUT=./videos
# Path where to save predictions
OUTPUT=./output
```

4. Run the following command to start the inference in Docker (make sure you are in the `inference_script` directory):

```bash
docker compose up
```

This will automatically pull our pre-built images `supervisely/mvd:inference-1.0.0` and `supervisely/mvd:rtdetrv2-1.0.0` and load your MVD model and the mouse detector in separate containers. After this, the inference will start. This process may take some time. The final predictions will be saved in `inference_script/output` (in case you didn't change the default output path).