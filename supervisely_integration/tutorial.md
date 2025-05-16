# How to Train Mouse Action Recognition Model

Table of Contents:
1. [Import Data](#1-import-data)
2. [Annotation](#2-annotation)
3. [Preprocessing](#3-preprocessing)
4. [Train & Evaluation](#4-train--evaluation)
5. [Inference](#5-inference)

## 1. Import Data

First, you need to import your data into Supervisely.

🔴🔴🔴
Create a project named "Input Project". Import your video data into the project.

When you get new data, import it into this same project. We recommend importing all new data into one project, since this way our algorithms will help optimize the next steps, such as preprocessing and training, to avoid doing work that has already been done.


## 2. Annotation

You may annotate your data in Supervisely labeling tool or import your annotations in CSV format.

The final annotation should be in the Supervisely format. Your annotations should contain the labels of the mouse actions, such as "Head/Body TWITCH" and "Self-Grooming" and be labeled as Tags. Each tag has a start time and end time, where the action is happening. You don't need to manually annotate bounding boxes around the mouse, because our mouse detector will do it at the preprocessing stage.

## 3. Preprocessing

When you have your data and annotations ready, you can start preprocessing your data for training. The preprocessing is done by the **Preprocess Data for Mouse Action Recognition** app in Supervisely and includes the following procedures:

1. **Mouse detection**. We use a separate mouse detector that predicts bounding boxes around the mouse in each frame. We then crop videos to the bounding box of the mouse. This is done to reduce the amount of background noise and focus on the mouse itself, which helps the model learn better and more efficiently.
2. **Trim videos into segments**. Videos are trimmed into short clips (~2-3 seconds each), which are manageable for the model. Each clip represents a particular mouse action. This is necessary because the MVD model has a context window limitation, it can't process large videos with a length of several minutes.
3. **Class balancing**. Since for most of the video frames the mouse performs no actions, there will be too many video clips with inactivity in the training set (we calle this "idle" action). To avoid this, we will balance the number of frames between the "idle" and "Self-Grooming" actions. This will create a more useful and informative sample for training the model.
4. **Train/test split**. The preprocessing app will create a test dataset with full-length original videos for evaluation purposes. This is important to evaluate the model performance on unseen data after training.

### How to Preprocess Data

1. Deploy a mouse detector. Run the app **Serve RT-DETRv2** in Supervisely and deploy our custom model trained for mouse detection task.
2. Run **Preprocess Data for Mouse Action Recognition** app in Supervisely, selecting the input project with your original videos and annotations. The input project may have a free structure with nested datasets, or without it.
3. Follow the instructions in the app. You will need to select the mouse detector you deployed in the first step, and specify the amount of video to train/test split. About 10 full-length videos should be enough for the test dataset.
4. Run the preprocessing.

After processing completes, a new project will be created with name **Training Data**. It will contain short video clips with detected bounding boxes of mice. It will consists of three datasets, one dataset per class: **"Self-Grooming"**, **"Head-Body_TWITCH"**, and **"idle"**. Additionally, the project will has a **test** dataset with full-length original videos for evaluation purposes.

> **Note**: The app will remember which videos it has already processed, so you can run it multiple times without reprocessing the same videos. This is useful if you add new videos to the input project - the app will only process the new videos and add them to the **Training Data** project that it has already created.


## 4. Train & Evaluation

After preprocessing, you can start training the model. The app **Train Mouse Action Recognition Model** in Supervisely will help you with this. It will train the [MVD](https://github.com/ruiwang2021/mvd) model for mouse action recognition.

### How to Train

To start training, run the app **Train Mouse Action Recognition Model** in Supervisely and select the **Training Data** project that had been created after preprocessing. You don't need the mouse detector for training, so you can stop the app **Serve RT-DETRv2** that you used for preprocessing to save GPU resources. Follow the instructions in the training app, specify the training parameters, and run the training. The training may take several hours or even days, depending on the amount of data and hyperparameters you choose. You can monitor the training process in the app.

**Notes on hyperparameters:**

- The MVD's archeticture is able to process only short video segments. The default input length is 16 frames. You can increase it by modifying `num_frames` parameter in the app's settings, but it will require more GPU memory.
- The `sampling_rate` is the second parameter that affects the context window length. It defines the step size between frames in the input video. For example, if `sampling_rate=2`, the model will process every second frame. This parameter affects the resulting model's performance, especially on the **Head/Body TWITCH** action, which is fast and may be missed if the sampling rate is too high. For better results, we recommend set `sampling_rate=1` and `num_frames=32`, but it may require high-end GPU, or multi-GPU setup.
- The resulting context window length is `num_frames * sampling_rate`. With default settings it is equal to 32 frames *(16 * 2 = 32)*, which is about 1 second of video.
- The `batch_size` parameter is also important. It should be at least 6-8 in case of single GPU setup.


### Evaluation

After training, the app will automatically evaluate the best checkpoint on the **test** dataset and provide you with Evaluation Report. The report will include the metrics such as precision, recall, and F1-score and will help you understand how well the model performs on unseen data.

## 5. Inference

Now, you can use the trained model for inference. As long as MVD is limited to short video segments, the model processes long videos in a sliding window manner. The model predicts an action for each short segment (window), and the app will aggregate the results to provide a final prediction for the entire video. The mouse detector is required for inference, it will be used to crop the video segments to the bounding box of the mouse.

### How to Run Inference

Run the app **Mouse Action Recognition** to start inference. It loads your trained model together with the mouse detector and make predictions on the input video project or dataset. The app will create a new project with the same structure as the input project, but with predictions for each video. The annotations will include the predicted bounding boxes of the mouse and the predicted action classes represented as tags. The inference may take several hours.

🔴🔴🔴TODO in inference app:
- сделать выбор датасета для инференса (не только проект)
- выбор модели MVD (checkpoint)
- запуск в докере
- выбор mouse detector - нужен ли? Нужно понять откуда он берется (из тим файлс? где должен лежать и тд)

### Inference in Docker

You can run inference in a Docker container with all dependencies pre-installed.

1. Clone github repository:

```bash
git clone https://github.com/supervisely-ecosystem/mouse-action-recognition
cd mouse-action-recognition
```

2. 🔴🔴🔴 Download your MVD model and the mouse detector. Place the models into `./models` directory.

3. Define the input video in the ENV file at `inference_script/.env`:

```bash
# Path to an input video or directory (relative to this .env file)
INPUT=/videos/GL010533.MP4
# Path where to save predictions
OUTPUT=../output
```

4. Run the following command to start the inference:

```bash
docker compose -f inference_script/docker-compose.yml up
```

This will load the MVD model and the mouse detector, and run inference on the input video. The predictions will be saved in the `output` directory.
