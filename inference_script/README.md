# How to run inference

1. Download the pretrained models of MVD and RT-DETRv2 using the following command:

```bash
bash inference_script/download_models.sh
```

The models will be saved in the `./models` directory.

2. Set your paths to an input video in `inference_script/.env`:

```bash
# Path to an input video or directory (relative to this .env file)
INPUT=/videos/GL010533.MP4
# Path where to save predictions
OUTPUT=../output
```

3. Run the following command to start the inference:

```bash
docker compose -f inference_script/docker-compose.yml up
```

The inference may take about 10 minutes depending on the video length.