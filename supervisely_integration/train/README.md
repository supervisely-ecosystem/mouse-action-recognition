<div align="center" markdown>

<img src=""/>  

# Train Mouse Action Recognition

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#how-to-run">How To Run</a> •
  <a href="#obtain-saved-checkpoints">Obtain saved checkpoints</a> •
  <a href="#how-to-use-your-checkpoints-outside-supervisely-platform">How to use checkpoints outside Supervisely Platform</a> •
  <a href="#acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/mouse-action-recognition/supervisely_integration/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/mouse-action-recognition)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/mouse-action-recognition/supervisely_integration/train.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/mouse-action-recognition/supervisely_integration/train.png)](https://supervisely.com)

</div>

# Overview

This app allows you to train models for mouse action recognition using the MVD architecture. The model can classify different mouse behaviors in video sequences such as "Self-Grooming" and "Head/Body TWITCH". The training uses a sliding window approach with maximal bounding box detection to focus on the mouse in each frame.

The app supports configurable hyperparameters, train/validation splits, and includes comprehensive evaluation metrics for model performance assessment.

## Action Classes

The model is trained to recognize the following action classes:

- **idle**: Mouse is not performing any specific action of interest
- **Self-Grooming**: Mouse is grooming itself 
- **Head/Body TWITCH**: Mouse exhibits quick, jerky movements of the head or body

# How to Run

**Step 0.** Run the app from context menu of the project with video annotations or from the Ecosystem

**Step 1.** Select if you want to use cached project or redownload it

![train-step-1]()

**Step 2.** Configure hyperparameters for the training process

Hyperparameters include:
- Training duration (epochs, checkpoint frequency)
- Dataset validation ratio
- Optimization parameters (learning rate, batch size, etc.)
- Model input configuration (input size, frames, sampling rate)
- Augmentation options

![train-step-2](https://github.com/supervisely-ecosystem/mouse-action-recognition/releases/download/v0.0.1/train-step-2.png)

**Step 3.** Enter experiment name and start training

![train-step-3](https://github.com/supervisely-ecosystem/mouse-action-recognition/releases/download/v0.0.1/train-step-3.png)

**Step 4.** Monitor training progress and metrics

![train-step-4](https://github.com/supervisely-ecosystem/mouse-action-recognition/releases/download/v0.0.1/train-step-4.png)

**Step 5.** Review benchmark results after training completes

The app automatically evaluates model performance on test data, providing metrics like precision, recall, and F1 score for each action class.

![train-step-5](https://github.com/supervisely-ecosystem/mouse-action-recognition/releases/download/v0.0.1/train-step-5.png)

# Obtain saved checkpoints

All trained checkpoints that are generated through the training process are stored in [Team Files](https://app.supervisely.com/files/) in the **experiments** folder.

You will see a folder thumbnail with a link to your saved checkpoints by the end of training process.

![checkpoints-location](https://github.com/supervisely-ecosystem/mouse-action-recognition/releases/download/v0.0.1/checkpoints-location.png)

# How to use your checkpoints outside Supervisely Platform

After you've trained a model in Supervisely, you can download the checkpoint from Team Files and use it as a PyTorch model without Supervisely Platform.

**Quick start:**

1. **Set up environment**. Install required packages for the MVD model.
2. **Download** your checkpoint from Supervisely Platform.
3. **Run inference**. Use the checkpoint to perform inference on new video data.

## Inference code example

```python
import torch
from src.inference.maximal_bbox_sliding_window import MaximalBBoxSlidingWindow3
from supervisely_integration.train.scripts.load import load_mvd
from supervisely_integration.train.scripts.predictor import predict_video

# Load model
checkpoint_path = "path/to/checkpoint.pth"
config_path = "path/to/config.txt"
model_meta = "path/to/model_meta.json"

model, opts = load_mvd(checkpoint_path, config_path)

# Run inference on video
video_path = "path/to/video.mp4"
predictions = predict_video(video_path, model, opts, model_meta, stride=8)

# Process predictions
for prediction in predictions:
    frame_range = prediction["frame_range"]
    label = prediction["label"]
    confidence = prediction["confidence"]
    print(f"Frames {frame_range}: Action {label} (confidence: {confidence:.2f})")
```

# Acknowledgment

This app is based on the `MVD` model ([github](https://github.com/ruiwang2021/mvd)). ![GitHub Org's stars](https://img.shields.io/github/stars/ruiwang2021/mvd?style=social)