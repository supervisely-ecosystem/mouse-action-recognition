import torch
from video_sliding_window_optimized import VideoSlidingWindow, pre_extract_frames
from frame_based_dataset import FrameBasedSlidingWindow
from torch.utils.data import DataLoader

# Method 1: Using optimized PyAV-based loading with caching
video_dataset = VideoSlidingWindow(
    video_path="/path/to/video.mp4",
    num_frames=16,
    frame_sample_rate=2,
    stride=5,
    cache_size=64
)

# Use with DataLoader for parallel processing
loader = DataLoader(
    video_dataset,
    batch_size=8,
    num_workers=4
)

# Method 2: Pre-extract frames and then use frame-based dataset
# frames_dir = "/path/to/extracted_frames"
# pre_extract_frames("/path/to/video.mp4", frames_dir)
# 
# frame_dataset = FrameBasedSlidingWindow(
#     frames_dir=frames_dir,
#     num_frames=16,
#     frame_sample_rate=2,
#     stride=5
# )
# 
# frame_loader = DataLoader(
#     frame_dataset,
#     batch_size=8,
#     num_workers=4
# )
