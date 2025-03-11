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
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
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


def save_frames_as_image(frames, output_filename="tmp/output.png"):
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

    # make grid
    nrows = 4
    ncols = int(np.ceil(len(frames) / nrows))
    
    # Get frame dimensions
    frame_height, frame_width = frames[0].shape[:2] if isinstance(frames, list) else frames.shape[1:3]
    
    # Create a blank canvas for the grid
    grid_height = nrows * frame_height
    grid_width = ncols * frame_width
    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Add timestamps based on fps
    timestamp_size = 1
    font_scale = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Place frames in the grid
    for i, frame in enumerate(frames if isinstance(frames, list) else frames):
        if i >= nrows * ncols:
            break
            
        row = i // ncols
        col = i % ncols
        
        y_start = row * frame_height
        y_end = y_start + frame_height
        x_start = col * frame_width
        x_end = x_start + frame_width
        
        # Copy the frame to the grid
        if isinstance(frames, list):
            grid_image[y_start:y_end, x_start:x_end] = frame
        else:
            grid_image[y_start:y_end, x_start:x_end] = frames[i]
        
        # Add timestamp text
        text = f"{i}"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = x_start + 10
        text_y = y_start + 20
        
        # Add a background rectangle for better readability
        cv2.rectangle(
            grid_image, 
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 0, 0),
            -1
        )
        
        # Add the text
        cv2.putText(
            grid_image,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    # Save the grid image to disk
    cv2.imwrite(output_filename, cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
    
    return grid_image