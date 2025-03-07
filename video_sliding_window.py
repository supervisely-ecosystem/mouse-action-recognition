import os
import numpy as np
import torch
import decord
from PIL import Image
from torchvision import transforms
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset, IterableDataset
import video_transforms as video_transforms
import volume_transforms as volume_transforms


class VideoSlidingWindow(IterableDataset):

    def __init__(self, video_path, num_frames=16,
                 frame_sample_rate=2, input_size=224,
                 stride=5):
        self.video_path = video_path
        self.num_frames = num_frames
        self.frame_sample_rate = frame_sample_rate  # step of frame sampling
        self.input_size = input_size
        self.stride = stride
        
        self.vr = VideoReader(video_path)
        # self.vr = VideoReader(video_path, width=self.new_width, height=self.new_height, num_threads=1, ctx=cpu(0))
            
        self.data_transform = video_transforms.Compose([
            video_transforms.Resize(size=(self.input_size, self.input_size), interpolation='bilinear'),
            # video_transforms.CenterCrop(size=(self.input_size, self.input_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        ])

    def __iter__(self):
        total_frames = len(self.vr)
        # Calculate effective frame indices considering sample rate
        effective_length = total_frames // self.frame_sample_rate
        
        # Determine how many complete windows we can extract
        num_windows = max(0, (effective_length - self.num_frames) // self.stride + 1)
        
        for window_idx in range(num_windows):
            start_idx = window_idx * self.stride
            end_idx = start_idx + self.num_frames
            
            # Get the actual frame indices considering the sample rate
            frame_indices = [i * self.frame_sample_rate for i in range(start_idx, end_idx)]
            
            # Handle potential out-of-bounds indices
            if frame_indices[-1] >= total_frames:
                break
                
            # Extract frames and process the window
            buffer = self._extract_frames(frame_indices)
            yield buffer

    def _extract_frames(self, frame_indices):
        # Extract frames at the specified indices
        buffer = self.vr.get_batch(frame_indices).asnumpy()
        buffer = self.data_transform(buffer)
        return buffer

    def __len__(self):
        total_frames = len(self.vr)
        effective_length = total_frames // self.frame_sample_rate
        num_windows = max(0, (effective_length - self.num_frames) // self.stride + 1)
        return num_windows        
