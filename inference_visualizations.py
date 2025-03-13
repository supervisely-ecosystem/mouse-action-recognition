import subprocess
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from bbox_utils import get_maximal_bbox

def draw_timeline(predictions, fps, experiment_name=None, class_names=None, figsize=(15, 7), output_dir='results'):
    """
    Draw a timeline of class probabilities from video predictions.
    
    Args:
        predictions: List of dictionaries containing predictions for each window
        fps: Frames per second of the video
        experiment_name: Name of the experiment for saving the visualization
        num_classes: Number of classes to visualize
        class_names: Optional list of class names for the legend
        figsize: Figure size (width, height) in inches
    
    Returns:
        The matplotlib figure
    """
    # Extract time and probability information
    times = []
    # Determine the actual number of classes from the first prediction
    num_classes = len(predictions[0]['probabilities'])
    
    all_probs = np.zeros((len(predictions), num_classes))
    
    for i, pred in enumerate(predictions):
        # Use midpoint of frame range as the time point
        frame_range = pred['frame_range']
        mid_frame = (frame_range[0] + frame_range[1]) / 2
        time_sec = mid_frame / fps
        times.append(time_sec)
        
        # Extract probabilities for each class
        probabilities = pred['probabilities']
        all_probs[i, :] = probabilities
    
    # Create the visualization
    fig = plt.figure(figsize=figsize)
    
    for idx in range(num_classes):
        label = class_names[idx] if class_names else f"Class {idx}"
        plt.plot(times, all_probs[:, idx], linewidth=2, label=label)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Probability')
    plt.title('Class Probabilities Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the visualization
    os.makedirs(output_dir, exist_ok=True)
    vis_path = f'{output_dir}/timeline_{experiment_name or "test"}.png'
    plt.savefig(vis_path)
    print(f"Visualization saved to {vis_path}")
    return fig

def write_positive_fragments(predictions, video_path, crop=False, output_dir='results'):
    """
    Creates a bunch of video clips of positive predictions. Prediction is positive if the label is not 0.
    The function creates a directory of video segments copied from the original video.
    Each segment is named according to the label and the frame range.
    For example, if the label is 1 and the frame range is [0, 32], the segment will be named "clip_0_32_1.mp4".
    The output directory will be created and will has a name based on the video name.
    It will be created in the `output_dir` directory.
    Use ffmpeg to create the segments.
    
    Adjacent fragments with the same label are merged into a single clip.
    
    The predictions has the following format:
    predictions = [{
        'frame_range': [0, 32],
        'label': 1,
        'confidence': 0.9613,
        'probabilities': [0.0387, 0.9613, 0.0000],
    }]
    """
    # Create output directory based on video name
    video_name = Path(video_path).stem
    clips_dir = Path(output_dir) / f"{video_name}_clips"
    os.makedirs(clips_dir, exist_ok=False)
    
    # Get video FPS using ffprobe
    probe_cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-select_streams', 'v:0', 
        '-show_entries', 'stream=r_frame_rate', 
        '-of', 'csv=p=0', 
        video_path
    ]
    
    result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    fps_str = result.stdout.strip()
    
    # Parse framerate (might be in format "30000/1001")
    if '/' in fps_str:
        num, den = map(int, fps_str.split('/'))
        fps = num / den
    else:
        fps = float(fps_str)
    
    print(f"Video FPS: {fps}")
    
    # Filter positive predictions (label != 0)
    positive_predictions = [pred for pred in predictions if pred['label'] != 0]
    print(f"Found {len(positive_predictions)} positive predictions out of {len(predictions)} total")
    
    if not positive_predictions:
        print("No positive predictions found.")
        return clips_dir
        
    # Sort positive predictions by start frame
    positive_predictions.sort(key=lambda x: x['frame_range'][0])
    
    # Merge overlapping or adjacent segments with the same label
    merged_segments = []
    
    for pred in positive_predictions:
        start_frame, end_frame = pred['frame_range']
        label = pred['label']
        confidence = pred['confidence']
        bbox = pred.get('maximal_bbox')
        
        # If this is the first segment or it doesn't overlap with previous segment
        if not merged_segments or start_frame > merged_segments[-1]['frame_range'][1] or label != merged_segments[-1]['label']:
            merged_segments.append({
                'frame_range': [start_frame, end_frame],
                'label': label,
                'confidence': confidence,
                'bbox': bbox,
            })
        else:
            # Extend the previous segment
            merged_segments[-1]['frame_range'][1] = max(merged_segments[-1]['frame_range'][1], end_frame)
            # Update confidence to the max of the two segments
            merged_segments[-1]['confidence'] = max(merged_segments[-1]['confidence'], confidence)
            # Update bbox if needed
            if crop:
                assert bbox is not None, "Bbox is required for cropping"
                merged_segments[-1]['bbox'] = get_maximal_bbox(
                    [merged_segments[-1]['bbox'], bbox]
                )
    
    print(f"Merged {len(positive_predictions)} positive predictions into {len(merged_segments)} segments")
    
    # Process each merged segment
    for i, segment in enumerate(tqdm(merged_segments)):
        start_frame, end_frame = segment['frame_range']
        label = segment['label']
        
        # Calculate time positions
        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps
        
        # Output file path
        output_file = clips_dir / f"clip_{start_frame}_{end_frame}_{label}.mp4"

        if crop:
            # ffmpeg command to extract the segment with cropping
            x1, y1, x2, y2 = segment['bbox']
            width = x2 - x1
            height = y2 - y1
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-ss', f"{start_time}",  # Start position
                '-i', video_path,  # Input file
                '-t', f"{duration}",  # Duration
                '-vf', f"crop={width}:{height}:{x1}:{y1}",  # Crop filter
                '-c:v', 'libx264',  # We can't use copy with filters
                '-preset', 'veryfast',  # Fastest encoding
                '-crf', '23',  # Quality factor (lower is better)
                '-pix_fmt', 'yuv420p',  # Pixel format
                '-an',  # Remove audio
                str(output_file)
            ]
        else:
            # ffmpeg command to extract the segment without re-encoding
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-ss', f"{start_time}",  # Start position
                '-i', video_path,  # Input file
                '-t', f"{duration}",  # Duration
                '-c:v', 'copy',  # Copy video stream without re-encoding (much faster)
                '-an',  # Remove audio
                str(output_file)
            ]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    print(f"All merged fragments saved to {clips_dir}")
    return clips_dir


if __name__ == "__main__":
    import json
    # Example usage
    video_path = "/root/volume/data/mouse/HOM Mice F.2632_HOM 12 Days post tre/12 Days post tre/video/GL010560.MP4"
    predictions_path = "results/predictions_MP_TRAIN_3_maximal_crop_2025-03-11_15-09-26.json"
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    write_positive_fragments(predictions, video_path)