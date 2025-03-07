import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import video_transforms as video_transforms
import volume_transforms as volume_transforms

class FrameBasedSlidingWindow(Dataset):
    """
    Dataset that works with pre-extracted frames saved on disk.
    More efficient for repeated access than decoding videos every time.
    """
    def __init__(self, frames_dir, num_frames=16, 
                 frame_sample_rate=2, input_size=224,
                 stride=5):
        self.frames_dir = frames_dir
        self.num_frames = num_frames
        self.frame_sample_rate = frame_sample_rate
        self.input_size = input_size
        self.stride = stride
        
        # Get sorted list of frame files
        self.frame_files = sorted([f for f in os.listdir(frames_dir) 
                                  if f.startswith("frame_") and f.endswith(".jpg")])
        
        self.data_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        frame_indices = [start_idx + i for i in range(self.num_frames)]
        frame_indices = [i * self.frame_sample_rate for i in frame_indices]
        
        # Load frames
        buffer = []
        for frame_idx in frame_indices:
            if frame_idx >= len(self.frame_files):
                # Handle boundary conditions by duplicating the last frame
                img_path = os.path.join(self.frames_dir, self.frame_files[-1])
            else:
                img_path = os.path.join(self.frames_dir, self.frame_files[frame_idx])
                
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.data_transform(img)
            buffer.append(img_tensor)
            
        # Stack to create batch
        buffer = torch.stack(buffer)  # Shape: [num_frames, C, H, W]
        return buffer
        
    def __len__(self):
        effective_length = len(self.frame_files) // self.frame_sample_rate
        return max(0, (effective_length - self.num_frames) // self.stride + 1)
