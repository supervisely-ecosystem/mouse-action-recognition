import os
import numpy as np
import torch
import av
from PIL import Image
from torchvision import transforms
import warnings
from torch.utils.data import Dataset, IterableDataset
import video_transforms as video_transforms
import volume_transforms as volume_transforms
from functools import lru_cache
import multiprocessing

class VideoSlidingWindow(IterableDataset):

    def __init__(self, video_path, num_frames=16,
                 frame_sample_rate=2, input_size=224,
                 stride=5, cache_size=32, num_workers=None):
        self.video_path = video_path
        self.num_frames = num_frames
        self.frame_sample_rate = frame_sample_rate
        self.input_size = input_size
        self.stride = stride
        self.cache_size = cache_size
        self.num_workers = num_workers if num_workers else max(1, multiprocessing.cpu_count() // 2)
        
        # Open the video file to get metadata
        container = av.open(video_path)
        self.stream = container.streams.video[0]
        self.total_frames = self.stream.frames
        self.fps = self.stream.average_rate
        container.close()
        
        # Initialize frame cache
        self._frame_cache = {}
        
        self.data_transform = video_transforms.Compose([
            video_transforms.Resize(size=(self.input_size, self.input_size), interpolation='bilinear'),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        ])
        
        # For multi-process loading support
        self.worker_id = 0
        self.num_workers = 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        
        effective_length = self.total_frames // self.frame_sample_rate
        num_windows = max(0, (effective_length - self.num_frames) // self.stride + 1)
        
        # Distribute windows among workers if using multi-process loading
        for window_idx in range(self.worker_id, num_windows, max(1, self.num_workers)):
            start_idx = window_idx * self.stride
            end_idx = start_idx + self.num_frames
            
            # Get the actual frame indices considering the sample rate
            frame_indices = [i * self.frame_sample_rate for i in range(start_idx, end_idx)]
            
            # Handle potential out-of-bounds indices
            if frame_indices[-1] >= self.total_frames:
                break
                
            # Extract frames and process the window
            buffer = self._extract_frames(frame_indices)
            yield buffer

    def _extract_frames(self, frame_indices):
        # Check if we have these frames in cache
        cache_key = tuple(frame_indices)
        if cache_key in self._frame_cache:
            return self._frame_cache[cache_key]
        
        # Extract frames using PyAV
        buffer = []
        container = av.open(self.video_path)
        stream = container.streams.video[0]
        
        # Set stream parameters for more efficient seeking
        stream.thread_type = "AUTO"
        stream.thread_count = self.num_workers
        
        for frame_idx in frame_indices:
            # Seek to the desired frame
            container.seek(frame_idx, stream=stream)
            for frame in container.decode(stream):
                img = frame.to_ndarray(format="rgb24")
                buffer.append(img)
                break  # Just take one frame
        
        container.close()
        
        # Convert to numpy array and apply transformations
        buffer = np.array(buffer)
        buffer = self.data_transform(buffer)
        
        # Cache the result if cache is enabled
        if self.cache_size > 0:
            # If cache is full, remove an item
            if len(self._frame_cache) >= self.cache_size:
                self._frame_cache.pop(next(iter(self._frame_cache)))
            self._frame_cache[cache_key] = buffer
            
        return buffer

    def __len__(self):
        effective_length = self.total_frames // self.frame_sample_rate
        num_windows = max(0, (effective_length - self.num_frames) // self.stride + 1)
        return num_windows

# Utility function for pre-extracting frames
def pre_extract_frames(video_path, output_dir):
    """
    Pre-extract frames from a video and save to disk for faster loading.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    container = av.open(video_path)
    stream = container.streams.video[0]
    
    for i, frame in enumerate(container.decode(stream)):
        img = frame.to_image()
        img.save(os.path.join(output_dir, f"frame_{i:06d}.jpg"))
    
    container.close()
    return i + 1  # Total number of frames
