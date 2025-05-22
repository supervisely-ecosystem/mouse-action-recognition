# How to Train Mouse Action Recognition Model

Table of Contents:
1. [Import Data](#1-import-data)
2. [Annotation](#2-annotation)
3. [Preprocessing](#3-preprocessing)
4. [Train & Evaluation](#4-train--evaluation)
5. [Inference](#5-inference)

## 1. Import Data

### How to import video files

To import video files from your local PC, you can follow these steps:

1. Create a new project in Supervisely, name it **Input Project**. If your workspace is empty, you can just click the **Import Data** button.
2. Choose the **Videos** option in the **Type of project** section, and click **Create**.
3. Next, you can drag and drop your video files into the project.

If you need to import files from a remote server or from a Cloud Storage, you can use apps like [Import Videos from Cloud Storage](https://ecosystem.supervisely.com/apps/import-videos-from-cloud-storage) or [Remote Import](https://ecosystem.supervisely.com/apps/remote-import). Check the [documentation](https://docs.supervisely.com/getting-started/how-to-import) for more details on how to import data.

**Note**: When you get new data, import it into the same project (**Input Project**). We recommend importing all new data into one project, since this way our algorithms will help optimize the next steps, such as preprocessing and training to avoid doing work that has already been done.

### Import CSV annotations

If you need to upload your annotations in CSV format, you can use our script from this [repository](https://github.com/supervisely-ecosystem/mouse-tests).


## 2. Annotation

Before training a NN model, you need to annotate your **Input Project**. We recommend to annotate your videos in Supervisely, because this way you can annotate precise frames where an action starts and ends, which will help the model learn much better. Also, the labeling tool is very convenient and has a lot of features that will help you annotate your data faster.

**Annotation format**: The annotations should contain the mouse actions (Head/Body Twitch, Self-Grooming) represented as **Tags**. Each tag has a start time and end time, where the action is happening. You don't need to manually annotate bounding boxes around the mouse, because our mouse detector will do it at the preprocessing stage.


## 3. Preprocessing

When you have your data and annotations ready, you can start preprocessing your data for training. The preprocessing is done by the **[Preprocess Data for Mouse Action Recognition](https://ecosystem.supervisely.com/apps/preprocess-data-for-mouse-project)** app in Supervisely and includes the following procedures:

1. **Mouse detection**. We use a separate mouse detector that predicts bounding boxes around the mouse in each frame. We then crop videos to the bounding box of the mouse. This is done to reduce the amount of background noise and focus on the mouse itself, which helps the model learn better and more efficiently.
2. **Trim videos into segments**. Videos are trimmed into short clips (~2-3 seconds each), which are manageable for the model. Each clip represents a particular mouse action. This is necessary because the MVD model has a context window limitation, it can't process large videos with a length of several minutes.
3. **Class balancing**. Since for most of the video frames the mouse performs no actions, there will be too many video clips with inactivity in the training set (we calle this "idle" action). To avoid this, we will balance the number of frames between the "idle" and "Self-Grooming" actions. This will create a more useful and informative sample for training the model.
4. **Train/test split**. The preprocessing app will create a test dataset with full-length original videos for evaluation purposes. This is important to evaluate the model performance on unseen data after training.

### How to Preprocess Data

1. Deploy a mouse detector. Run the app **[Serve RT-DETRv2](https://ecosystem.supervisely.com/apps/rt-detrv2/supervisely_integration/serve)** in Supervisely and deploy our custom model trained for mouse detection task.
2. Run **[Preprocess Data for Mouse Action Recognition](https://ecosystem.supervisely.com/apps/preprocess-data-for-mouse-project)** app in Supervisely, selecting the input project with your original videos and annotations. The input project may have a free structure with nested datasets, or without it.
3. Follow the instructions in the app. You will need to select the mouse detector you deployed in the first step, and specify the amount of video to train/test split. About 10 full-length videos should be enough for the test dataset.
4. Run the preprocessing.

After processing completes, a new project will be created with name **Training Data**. It will contain short video clips with detected bounding boxes of mice. It will consists of three datasets, one dataset per class: **"Self-Grooming"**, **"Head-Body_TWITCH"**, and **"idle"**. Additionally, the project will has a **test** dataset with full-length original videos for evaluation purposes.

**Note**: The app will remember which videos it has already processed, so you can run it multiple times without reprocessing the same videos. This is useful if you add new videos to the input project - the app will only process the new videos and add them to the **Training Data** project that it has already created.


## 4. Train & Evaluation

After preprocessing, you can start training the model. The app **[Train Mouse Action Recognition Model](https://ecosystem.supervisely.com/apps/mouse-action-recognition/supervisely_integration/train)** in Supervisely will help you with this. It will train the [MVD](https://github.com/ruiwang2021/mvd) model for mouse action recognition.

### How to Train

To start training, run the app **[Train Mouse Action Recognition Model](https://ecosystem.supervisely.com/apps/mouse-action-recognition/supervisely_integration/train)** in Supervisely and select the **Training Data** project that had been created after preprocessing. You don't need the mouse detector for training, so you can stop the app **Serve RT-DETRv2** that you used for preprocessing to save GPU resources. Follow the instructions in the training app, specify the training parameters, and run the training. The training may take several hours or even days, depending on the amount of data and hyperparameters you choose. You can monitor the training process in the app.

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

Launch the app **[Mouse Action Recognition](https://ecosystem.supervisely.com/apps/mouse-action-recognition/src/app)** providing the paths to your trained MVD model and the mouse detector in Team Files. The app will load the models and run inference on the input project or dataset. After the inference is done, it will create a new project with the same structure as the input project, but with predictions for each video. The annotations will include predicted bounding boxes of a mouse and action intervals. The predicted action classes are represented as tags on the timeline in the labeling tool. The inference may take several hours for long-video datasets.

### Inference in Docker

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