import os
import numpy as np
import cv2
from PIL import Image
import torch
import tempfile
import subprocess


def save_frames_as_video(frames, output_filename="tmp/output.mp4", fps=30):
    # shape (16, 320, 566, 3) : (N, H, W, C)
    if isinstance(frames, list) and isinstance(frames[0], Image.Image):
        frames = [np.array(frame) for frame in frames]
        frames = np.stack(frames)
    elif isinstance(frames, torch.Tensor):
        # (3, 16, 320, 566) -> (16, 320, 566, 3)
        frames = frames.permute(1, 2, 3, 0).cpu().numpy()
        # normalize to 0-255. Tensor is in ImageNet format
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        frames = frames * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
    else:
        assert isinstance(frames, np.ndarray), "frames must be a list of numpy arrays or a numpy array"

    # Create a temporary directory to store frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save each frame as a PNG file
        for i in range(frames.shape[0]):
            frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
            cv2.imwrite(frame_path, cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
        
        # Use FFmpeg to create a video from the frames
        subprocess.call([
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-loglevel', 'error',  # Only show errors, suppress warnings and info
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-profile:v', 'high',
            '-crf', '17',  # Quality (lower is better)
            '-pix_fmt', 'yuv420p',  # Most compatible pixel format
            output_filename
        ])
    
    return output_filename
