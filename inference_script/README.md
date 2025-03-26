# How to run inference

1. Install docker and docker compose if not installed

2. Update following vars in `inference_script/.env`:
```bash
# Inference
# Path to an input directory or video
INPUT=/path/to/video.mp4
# Path where to save predictions
OUTPUT=../output
```

3. Run this command
`docker compose -f inference_script/docker-compose.yml up`

