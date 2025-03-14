import os
import random
import shutil
import glob
from pathlib import Path
import argparse

def find_video_paths(base_dir):
    """Find all video paths in the nested structure."""
    video_paths = []
    
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root) == "video":
            video_files = [os.path.join(root, f) for f in files if f.endswith('.MP4')]
            video_paths.extend(video_files)
    
    return video_paths

def get_annotation_path(video_path):
    """Get the corresponding annotation path for a video path."""
    # Given pattern:
    # video_path: .../datasets/.../video/GLXXXXXX.MP4
    # annotation_path: .../datasets/.../ann/GLXXXXXX.MP4.json
    
    video_dir = os.path.dirname(video_path)
    parent_dir = os.path.dirname(video_dir)
    
    video_filename = os.path.basename(video_path)
    ann_filename = f"{video_filename}.json"
    
    # Replace 'video' directory with 'ann' directory in the path
    ann_dir = os.path.join(parent_dir, "ann")
    ann_path = os.path.join(ann_dir, ann_filename)
    
    return ann_path

def sample_dataset(base_dir, output_dir, n_samples=50, seed=42):
    """Sample n_samples videos and their annotations from the dataset."""
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Find all video paths
    video_paths = find_video_paths(base_dir)
    
    if not video_paths:
        print(f"No videos found in {base_dir}")
        return
    
    # Sample n_samples videos (or all if there are fewer)
    n_samples = min(n_samples, len(video_paths))
    sampled_video_paths = random.sample(video_paths, n_samples)
    
    # Create output directories
    out_video_dir = os.path.join(output_dir, "videos")
    out_ann_dir = os.path.join(output_dir, "annotations")
    
    os.makedirs(out_video_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)
    
    # Copy sampled videos and their annotations
    copied_count = 0
    for video_path in sampled_video_paths:
        video_filename = os.path.basename(video_path)
        ann_path = get_annotation_path(video_path)
        
        # Check if annotation exists
        if not os.path.exists(ann_path):
            print(f"Warning: Annotation not found for {video_path}")
            continue
        
        # Copy video
        shutil.copy2(video_path, os.path.join(out_video_dir, video_filename))
        
        # Copy annotation
        ann_filename = os.path.basename(ann_path)
        shutil.copy2(ann_path, os.path.join(out_ann_dir, ann_filename))
        
        copied_count += 1
        
    print(f"Successfully copied {copied_count} videos and their annotations to {output_dir}")
    print(f"Output structure:\n{out_video_dir}\n{out_ann_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample videos and annotations from a dataset")
    parser.add_argument("--input", default="/root/volume/data/mouse/MP_TRAIN_3", 
                        help="Base input directory containing the dataset")
    parser.add_argument("--output", default="./sampled_dataset", 
                        help="Output directory for sampled data")
    parser.add_argument("--samples", type=int, default=100, 
                        help="Number of samples to extract")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print(f"Sampling {args.samples} videos from {args.input}")
    sample_dataset(args.input, args.output, args.samples, args.seed)