import os
import json
import math
import random
import subprocess
from pathlib import Path
import decord
from tqdm import tqdm

def main(
    video_file: str, 
    ann_file: str, 
    output_dir: str, 
    splits_file: str, 
    min_clip_duration: int = 3, 
    max_clip_duration: int = 5, 
    total_length_factor: float = 1.0
):
    video_path = Path(video_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open video with decord to fetch FPS and total frames.
    vr = decord.VideoReader(str(video_path))
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")

    # Read annotation text file and parse skip ranges.
    # Each line: clip_name start_frame end_frame
    skip_ranges = []
    total_skip_frames = 0
    with open(ann_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            s = int(parts[1])
            e = int(parts[2])
            skip_ranges.append([s, e])
            total_skip_frames += (e - s + 1)
    
    target_length = int(total_skip_frames * total_length_factor)

    # Merge overlapping skip ranges.
    skip_ranges = sorted(skip_ranges, key=lambda x: x[0])
    merged = []
    for rng in skip_ranges:
        if not merged or rng[0] > merged[-1][1] + 1:
            merged.append(rng)
        else:
            merged[-1][1] = max(merged[-1][1], rng[1])
    skip_ranges = merged

    # Build non-skip intervals from video frames.
    non_skip_intervals = []
    current = 0
    for s, e in skip_ranges:
        if current < s:
            non_skip_intervals.append([current, s-1])
        current = e + 1
    if current < total_frames:
        non_skip_intervals.append([current, total_frames-1])

    # Clip parameters in frames.
    clip_min_frames = math.ceil(fps * min_clip_duration)
    clip_max_frames = math.floor(fps * max_clip_duration)
    clip_counter = 1
    cumulative_clip_frames = 0
    info_lines = []

    # Process non-skip intervals and extract random clips.
    for interval in tqdm(non_skip_intervals):
        interval_start, interval_end = interval
        t = interval_start
        while t + clip_min_frames - 1 <= interval_end and cumulative_clip_frames < target_length:
            available = interval_end - t + 1
            if available < clip_min_frames:
                break
            clip_length = random.randint(clip_min_frames, min(clip_max_frames, available))
            start_frame = t
            end_frame = t + clip_length - 1
            duration = clip_length / fps
            clip_name = f"clip_{clip_counter:03d}.mp4"
            clip_path = out_dir / clip_name
            start_time = start_frame / fps
            # Extract clip using ffmpeg.
            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(video_path),
                "-ss", f"{start_time}",
                "-t", f"{duration}",
                # "-vf", "scale=224:224",  # Width:Height
                # "-c", "copy",
                # Remove the "-c copy" and add these parameters instead
                "-c:v", "libx264",  # Use H.264 codec
                "-preset", "medium",  # Balance between speed and quality
                "-crf", "18",      # Quality setting (lower = better quality)
                "-pix_fmt", "yuv420p",
                str(clip_path)
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Record clip info: clip_name, start_frame, end_frame, number_of_frames, duration
            info_lines.append(f"{clip_name} {start_frame} {end_frame} {clip_length} {duration:.2f}")
            cumulative_clip_frames += clip_length
            clip_counter += 1
            t = end_frame + 1
            if cumulative_clip_frames >= target_length:
                break

    # Write all clip info lines to splits.txt.
    with open(splits_file, "w") as f:
        for line in info_lines:
            f.write(line + "\n")

if __name__ == "__main__":
    video_file = "HOM Mice F.2632_HOM 12 Days post tre/12 Days post tre/video_320px/GL020560.mp4"
    clips_dir = "clips_negative_320px"
    info_file = "info_negative.txt"
    ann_file = "info.txt"  # positive frame ranges

    video_file = Path(video_file)
    ann_file = video_file.parent.parent / ann_file
    output_dir = video_file.parent.parent / f"{video_file.stem}-{clips_dir}"
    main(video_file, ann_file, output_dir, output_dir.parent / info_file)