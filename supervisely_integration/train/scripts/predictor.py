import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List
from torch.utils.data import DataLoader
from src.inference.maximal_bbox_sliding_window import MaximalBBoxSlidingWindow, MaximalBBoxSlidingWindow3
from supervisely.video_annotation.frame import Frame
from supervisely.video_annotation.video_annotation import VideoAnnotation
from supervisely.io.json import load_json_file
from supervisely.video_annotation.key_id_map import KeyIdMap


def get_video_ann(video_path, model_meta) -> List[Frame]:
    video_ann_path = Path(video_path).parent.parent / "ann" / (Path(video_path).name + ".json")
    video_ann_path = str(video_ann_path)
    ann_json = load_json_file(video_ann_path)
    video_ann = VideoAnnotation.from_json(ann_json, model_meta, KeyIdMap())
    return video_ann

def predict_video(video_path, model, opts, model_meta, stride, pbar=None):
    # Read the video

    video_ann = get_video_ann(video_path, model_meta)
    dataset = MaximalBBoxSlidingWindow3(
        video_path=video_path,
        video_ann=video_ann,
        num_frames=opts.num_frames,
        frame_sample_rate=opts.sampling_rate,
        input_size=opts.input_size,
        stride=stride,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=MaximalBBoxSlidingWindow.collate_fn,
    )

    # Inference
    device = opts.device
    print(f"Inference on {video_path}")
    print(f"dataset length: {len(dataset)}")

    predictions = []
    if pbar is None:
        iterator = tqdm(data_loader)
    else:
        iterator = data_loader
    last_notified = 0
    for input, frame_indices, bboxes in iterator:
        input = input.to(device)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output = model(input)

        # Get probabilities from the model output
        probs = torch.softmax(output, dim=1)
        for i, frame_idxs in enumerate(frame_indices):
            prob = probs[i].cpu().numpy()
            predicted_class = int(np.argmax(prob))
            confidence = float(prob[predicted_class])
            frame_range = [
                int(frame_idxs[0]),
                int(frame_idxs[-1]) + opts.sampling_rate,
            ]  # inclusive range
            bbox = bboxes[i]
            predictions.append(
                {
                    "frame_range": frame_range,
                    "label": predicted_class,
                    "confidence": confidence,
                    "probabilities": prob.tolist(),
                    "maximal_bbox": bbox,
                }
            )
        if pbar is not None:
            current_frame = frame_indices[-1][-1]
            count = current_frame - last_notified
            last_notified = current_frame
            pbar.update(count)

    return predictions
