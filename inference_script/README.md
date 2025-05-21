# How to run inference

1. Clone github repository, and navigate to the `inference_script` directory:

```bash
git clone https://github.com/supervisely-ecosystem/mouse-action-recognition
cd mouse-action-recognition/inference_script
```

2. Download your MVD model and the mouse detector:

```bash
bash download_models.sh
```

The models will be saved in the `inference_script/models` directory.

3. Open `.env` file and define your paths to an input video and MVD model. Here is an example of the `.env` file:

```bash
# Note: all paths are relative to this .env file (you can use absolute paths as well)
# Directory where the MVD model is stored
MODEL_DIR=models/MP_TRAIN_3_maximal_crop_2025-03-11_15-09-26
# Path to an input directory or a single video
INPUT=/volume/data/mouse/GL010533_sample.mp4
# Path where to save predictions
OUTPUT=output
```

4. Run the following command to start the inference in Docker (make sure you are in the `inference_script` directory):

```bash
docker compose up
```

This will load the MVD model and the mouse detector, and run inference on input videos. The inference may take some time. The final predictions will be saved in the `./output` directory. 