import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor

def analyze_annotations(annotation_path):
    """
    Analyze annotation file to check if all frames have at least one bounding box.
    Returns a tuple (is_valid, issues) where is_valid is a boolean and issues is a list of problematic frames.
    """
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    if len(annotation['frames']) != annotation['framesCount']:
        print(f"Warning: Frame count mismatch in {annotation_path}")
    
    issues = []
    for frame in annotation['frames']:
        frame_index = frame['index']
        num_figures = len(frame['figures'])
        
        if num_figures == 0:
            issues.append(f"Frame {frame_index} has no bounding boxes")
        elif num_figures > 1:
            issues.append(f"Frame {frame_index} has {num_figures} bounding boxes (will use the largest one)")
    
    return len(issues) == 0, issues

def get_bbox_area(bbox):
    """
    Calculate the area of a bounding box.
    """
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    return (x2 - x1) * (y2 - y1)

def get_largest_bbox(figures):
    """
    Get the largest bounding box from a list of figures.
    """
    largest_area = -1
    largest_bbox = None
    
    for figure in figures:
        bbox = figure['geometry']['points']['exterior']
        area = get_bbox_area(bbox)
        
        if area > largest_area:
            largest_area = area
            largest_bbox = bbox
    
    return largest_bbox

def get_square_bbox(bbox, padding=0.0):
    """
    Convert rectangular bbox to square by adding padding to the shortest edge.
    Adds additional padding to all sides.
    Returns (x, y, size) where (x, y) is the top-left corner and size is the side length.
    """
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    
    width = x2 - x1
    height = y2 - y1
    
    if width > height:
        # Add padding to height
        diff = width - height
        y1 = max(0, y1 - diff // 2)
        y2 = y1 + width
    else:
        # Add padding to width
        diff = height - width
        x1 = max(0, x1 - diff // 2)
        x2 = x1 + height
    
    # Add additional padding
    size = x2 - x1
    pad_px = padding * size // 2
    x1 = max(0, x1 - pad_px)
    y1 = max(0, y1 - pad_px)
    x2 = x2 + pad_px
    y2 = y2 + pad_px
    
    return int(x1), int(y1), int(x2 - x1)  # x, y, size

def process_video(video_path, annotation_path, output_path, target_size=320, pad_bbox=0.5):
    """
    Process a video by cropping frames based on bounding box annotations.
    """
    # Check if annotation is valid
    is_valid, issues = analyze_annotations(annotation_path)
    if not is_valid:
        print(f"Warning: {os.path.basename(video_path)} has invalid annotations:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Load annotations
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_size, target_size))
    
    # Process each frame
    frame_idx = 0
    last_valid_bbox = None
    
    # Create a progress bar
    with tqdm(total=min(frame_count, len(annotation['frames'])), 
              desc=f"Processing {os.path.basename(video_path)}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Find the corresponding annotation
            bbox = None
            for ann_frame in annotation['frames']:
                if ann_frame['index'] == frame_idx:
                    if len(ann_frame['figures']) > 0:
                        # If there are multiple bounding boxes, get the largest one
                        if len(ann_frame['figures']) > 1:
                            bbox = get_largest_bbox(ann_frame['figures'])
                        else:
                            bbox = ann_frame['figures'][0]['geometry']['points']['exterior']
                    break
            
            # If no bbox found for this frame, use the last valid one
            if bbox is None:
                bbox = last_valid_bbox
                if bbox is None:
                    print(f"Warning: No valid bbox found for frame {frame_idx} and no previous bbox available in {video_path}")
                    # Skip this frame
                    frame_idx += 1
                    pbar.update(1)
                    continue
            else:
                # Update the last valid bbox
                last_valid_bbox = bbox
            
            # Get square bbox
            x, y, size = get_square_bbox(bbox, padding=pad_bbox)
            
            # Ensure the bbox is within the frame boundaries
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            size = min(size, width - x, height - y)
            
            # Crop the frame
            cropped = frame[y:y+size, x:x+size]
            
            # Resize to target size
            resized = cv2.resize(cropped, (target_size, target_size))
            
            # Write to output
            out.write(resized)
            
            frame_idx += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    
    return True

def main():
    base_dir = "./data/mouse"
    input_dir = os.path.join(base_dir, "output")
    annotation_dir = os.path.join(base_dir, "detections")
    output_dir = os.path.join(base_dir, "cropped")
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if subdirectories exist
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    if not os.path.exists(annotation_dir):
        print(f"Error: Annotation directory {annotation_dir} does not exist")
        return
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    subdirs = sorted(subdirs)

    # Process all videos
    for subdir in subdirs:
        input_subdir = os.path.join(input_dir, subdir)
        annotation_subdir = os.path.join(annotation_dir, subdir)
        output_subdir = os.path.join(output_dir, subdir)
        
        # Create output subdirectory
        os.makedirs(output_subdir, exist_ok=True)
        
        # Get all videos in the subdirectory
        videos = [f for f in os.listdir(input_subdir) if f.lower().endswith('.mp4')]
        videos = sorted(videos)
        
        print(f"Processing {len(videos)} videos in subdirectory {subdir}...")
        
        # First, analyze all annotations
        print("Analyzing annotations...")
        invalid_annotations = []
        
        for video_name in videos:
            video_path = os.path.join(input_subdir, video_name)
            annotation_path = os.path.join(annotation_subdir, video_name + '.json')
            
            if not os.path.exists(annotation_path):
                print(f"Warning: No annotation file found for {annotation_path}")
                continue
            
            is_valid, issues = analyze_annotations(annotation_path)
            if not is_valid:
                invalid_annotations.append((video_name, issues))
        
        if invalid_annotations:
            print("\nThe following videos have issues with their annotations:")
            for video_name, issues in invalid_annotations:
                print(f"{video_name}:")
                for issue in issues:
                    print(f"  - {issue}")
            
            proceed = input("\nDo you want to proceed with processing these videos? (y/n): ")
            if proceed.lower() != 'y':
                print("Skipping this subdirectory.")
                continue
        
        # Process videos
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
            futures = []
            for video_name in videos:
                video_path = os.path.join(input_subdir, video_name)
                annotation_path = os.path.join(annotation_subdir, video_name + '.json')
                output_path = os.path.join(output_subdir, video_name)
                
                if not os.path.exists(annotation_path):
                    print(f"Warning: No annotation file found for {annotation_path}")
                    continue
                
                process_video(video_path, annotation_path, output_path)
                
            #     futures.append(executor.submit(
            #         process_video, video_path, annotation_path, output_path
            #     ))
            
            # # Wait for all tasks to complete
            # for future in futures:
            #     future.result()
        
        print(f"Finished processing subdirectory {subdir}")
    
    print("All videos processed successfully!")

if __name__ == "__main__":
    main()