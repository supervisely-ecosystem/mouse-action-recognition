<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/mouse-action-recognition/releases/download/media/poster_mouse_train-black.jpg">  

# Train Mouse Action Recognition

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#prerequisites">Prerequisites</a> •
  <a href="#how-to-run">How To Run</a> •
  <a href="#obtain-saved-checkpoints">Obtain saved checkpoints</a> •
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

# Prerequisites

Before using this training app, you must first prepare your video data using the **Preprocessing App**. The preprocessing app performs essential steps to make your data ready for training:

## Preprocessing App Features

- Creates a properly structured project with train and test datasets
- Splits videos into training and test sets
- Extracts training clips from labeled videos for three categories:
  - "Self-Grooming"
  - "Head-Body_TWITCH" 
  - "idle" (negative examples)
- Applies a mouse detection model to uploaded videos
- Maintains a cache to avoid reprocessing previously handled videos

## Preprocessing Output

After running the preprocessing app, you will get a new project with the name `[source project id] Training Data` containing:

- **train dataset** with 3 nested datasets of short video clips:
  - "Self-Grooming"
  - "Head-Body_TWITCH"
  - "idle"
- **test dataset** with full-length original videos for validation

**Important:** The Training App must be launched from this preprocessed project. Running the Training App on unprocessed video projects can result in an error.

# How to Run

**Step 0.** Run the app from context menu of the preprocessed project with video annotations

**Step 1.** Select if you want to use cached project or redownload it

![train-step-1](https://github.com/supervisely-ecosystem/mouse-action-recognition/releases/download/media/train-step-1.png)

**Step 2.** Select Model. Currently only 1 model is available

![train-step-2](https://github.com/supervisely-ecosystem/mouse-action-recognition/releases/download/media/train-step-2.png)

**Step 3.** Configure hyperparameters for the training process

Hyperparameters include:
- Training duration (epochs, checkpoint frequency)
- Dataset validation ratio
- Optimization parameters (learning rate, batch size, etc.)
- Model input configuration (input size, frames, sampling rate)
- Augmentation options

![train-step-3](https://github.com/supervisely-ecosystem/mouse-action-recognition/releases/download/media/train-step-3.png)

**Step 4.** Enter experiment name and start training

![train-step-4](https://github.com/supervisely-ecosystem/mouse-action-recognition/releases/download/media/train-step-4.png)

**Step 5.** Monitor training progress and metrics

![train-step-5](https://github.com/supervisely-ecosystem/mouse-action-recognition/releases/download/media/train-step-5.png)

**Step 6.** Review benchmark results after training completes

The app automatically evaluates model performance on test data, providing metrics like precision, recall, and F1 score for each action class.

![train-step-6](https://github.com/supervisely-ecosystem/mouse-action-recognition/releases/download/media/train-step-6.png)

# Obtain saved checkpoints

All trained checkpoints that are generated through the training process are stored in Team Files in the **experiments** folder.

You will see a folder thumbnail with a link to your saved checkpoints by the end of training process.

![artifacts-location](https://github.com/supervisely-ecosystem/mouse-action-recognition/releases/download/media/train-artifacts.png)

# Acknowledgment

This app is based on the `MVD` model ([github](https://github.com/ruiwang2021/mvd)). ![GitHub Org's stars](https://img.shields.io/github/stars/ruiwang2021/mvd?style=social)
