import subprocess
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from tqdm import tqdm
from src.bbox_utils import get_maximal_bbox


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


def draw_class_segments_timeline(predictions, fps, experiment_name=None, class_names=None, 
                                figsize=(15, 7), output_dir='results', min_segment_duration=0.0):
    """
    Draw a timeline of segments showing the most probable class at each time point.
    
    Args:
        predictions: List of dictionaries containing predictions for each window
        fps: Frames per second of the video
        experiment_name: Name of the experiment for saving the visualization
        class_names: Optional list of class names for the y-axis
        figsize: Figure size (width, height) in inches
        output_dir: Directory to save the visualization
        min_segment_duration: Minimum duration (in seconds) for a segment to be displayed
        
    Returns:
        The matplotlib figure
    """
    # Extract time and most probable class information
    times = []
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
    
    # Find the most probable class at each time point
    max_class_indices = np.argmax(all_probs, axis=1)
    
    # Set up the visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate color map based on the number of classes
    cmap = plt.cm.get_cmap('tab10', num_classes)
    
    # Find segments of continuous class predictions
    segments = []
    current_class = max_class_indices[0]
    segment_start = times[0]
    segment_confidence = all_probs[0, current_class]
    confidences = [segment_confidence]
    
    # Process each time point
    for i in range(1, len(times)):
        if max_class_indices[i] != current_class:
            # End of segment
            segment_end = times[i]
            segment_duration = segment_end - segment_start
            avg_confidence = np.mean(confidences)
            
            # Only add segment if it meets minimum duration
            if segment_duration >= min_segment_duration:
                segments.append({
                    'class': current_class,
                    'start_time': segment_start,
                    'end_time': segment_end,
                    'confidence': avg_confidence
                })
            
            # Start new segment
            current_class = max_class_indices[i]
            segment_start = times[i]
            confidences = [all_probs[i, current_class]]
        else:
            # Continue current segment
            confidences.append(all_probs[i, current_class])
    
    # Add the last segment
    segment_end = times[-1]
    segment_duration = segment_end - segment_start
    avg_confidence = np.mean(confidences)
    
    if segment_duration >= min_segment_duration:
        segments.append({
            'class': current_class,
            'start_time': segment_start,
            'end_time': segment_end,
            'confidence': avg_confidence
        })
    
    # Draw segments
    for segment in segments:
        class_idx = segment['class']
        color = cmap(class_idx)
        
        # Use transparency to represent confidence
        alpha = max(0.3, min(0.9, segment['confidence']))
        segment_color = to_rgba(color, alpha)
        
        # Create a rectangle for the segment
        rect = patches.Rectangle(
            (segment['start_time'], class_idx - 0.4),
            segment['end_time'] - segment['start_time'],
            0.8,
            linewidth=1,
            edgecolor=color,
            facecolor=segment_color,
            label=class_names[class_idx] if class_names else f"Class {class_idx}"
        )
        ax.add_patch(rect)
    
    # Set up the axes
    if class_names:
        plt.yticks(range(num_classes), class_names)
    else:
        plt.yticks(range(num_classes), [f"Class {i}" for i in range(num_classes)])
    
    # Set axis limits
    ax.set_xlim(min(times), max(times))
    ax.set_ylim(-0.5, num_classes - 0.5)
    
    # Add labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Class')
    plt.title('Most Probable Class Segments Over Time')
    plt.grid(True, alpha=0.2, linestyle='--')
    
    # Handle legend - get unique class entries
    handles, labels = [], []
    for class_idx in range(num_classes):
        if class_idx in [segment['class'] for segment in segments]:
            patch = patches.Patch(
                color=cmap(class_idx),
                label=class_names[class_idx] if class_names else f"Class {class_idx}"
            )
            handles.append(patch)
            labels.append(class_names[class_idx] if class_names else f"Class {class_idx}")
    
    plt.legend(handles=handles, labels=labels, loc='upper center', 
               bbox_to_anchor=(0.5, -0.1), ncol=min(5, len(handles)))
    
    plt.tight_layout()
    
    # Save the visualization
    os.makedirs(output_dir, exist_ok=True)
    vis_path = f'{output_dir}/class_segments_{experiment_name or "test"}.png'
    plt.savefig(vis_path, bbox_inches='tight')
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
    
    write_positive_fragments(predictions, video_path, crop=True, output_dir='results')