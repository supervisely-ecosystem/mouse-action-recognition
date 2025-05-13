from typing import List

from maximal_crop_dataset import get_maximal_bbox, get_square_bbox
from src.inference.video_sliding_window import VideoSlidingWindow
from supervisely.video_annotation.frame import Frame
from supervisely.video_annotation.video_annotation import VideoAnnotation


def get_maximal_bbox_crop(figures, img_size, padding=0.0):
    w, h = img_size
    if not figures:
        print(f"No found any detections")
        x1, y1, x2, y2 = 0, 0, w, h
    else:
        x1, y1, x2, y2 = get_maximal_bbox(figures)
        x1, y1, x2, y2 = get_square_bbox((x1, y1, x2, y2), padding=padding)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return x1, y1, x2, y2


class MaximalBBoxSlidingWindowTrainApp(VideoSlidingWindow):
    def __init__(
        self,
        video_path: str,
        video_ann: VideoAnnotation = None,
        num_frames: int = 16,
        frame_sample_rate: int = 2,
        input_size: int = 224,
        stride: int = 5,
        bbox_padding: float = 0.05,
    ):
        super().__init__(video_path, num_frames, frame_sample_rate, input_size, stride)
        self.bbox_padding = bbox_padding
        self.max_cache_size = 128
        self.buffer = []
        self.frame_index = 0
        self.frame_sample_rate = frame_sample_rate

        self.video_ann: VideoAnnotation = video_ann
        self.frames: List[Frame] = (
            [frame for frame in video_ann.frames] if video_ann is not None else []
        )

    def __iter__(self):
        frame_idx = 0
        total_frames = len(self.vr)

        # Fill the buffer to the desired size
        while len(self.buffer) < self.num_frames and frame_idx < len(self.frames):
            frame = self.frames[frame_idx]
            self.buffer.append(frame.to_json())
            frame_idx += 1

        # Main processing loop
        while (
            frame_idx < len(self.frames)
            and self.frame_index + (self.num_frames - 1) * self.frame_sample_rate
            < total_frames
        ):
            # Create a list of frame indices for the current window
            frame_indices = list(
                range(
                    self.frame_index,
                    self.frame_index + self.num_frames * self.frame_sample_rate,
                    self.frame_sample_rate,
                )
            )

            # Get the frame data and process it
            try:
                buffer = self.vr.get_batch(frame_indices).asnumpy()
                w, h = buffer.shape[2], buffer.shape[1]

                # Extract objects from the frame buffer
                figures = [fig for frame in self.buffer for fig in frame["figures"]]

                # Get the maximal bounding box
                x1, y1, x2, y2 = get_maximal_bbox_crop(
                    figures, (w, h), padding=self.bbox_padding
                )

                # Crop and transform the buffer
                buffer = buffer[:, y1:y2, x1:x2, :]
                buffer = self.data_transform(buffer)

                yield buffer, frame_indices, (x1, y1, x2, y2)

            except Exception as e:
                print(f"Error processing frames with indices {frame_indices}: {str(e)}")

            # Update the frame index and remove old frames from the buffer
            self.frame_index += self.stride * self.frame_sample_rate

            # Remove the first stride frames from the buffer
            for _ in range(self.stride):
                if self.buffer:
                    self.buffer.pop(0)

            # Add new frames to the buffer
            for _ in range(self.stride):
                if frame_idx < len(self.frames):
                    self.buffer.append(self.frames[frame_idx].to_json())
                    frame_idx += 1
