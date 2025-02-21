import os
import json
import math
import subprocess
from pathlib import Path
import decord
from tqdm import tqdm

def get_frame_ranges(ann_file: dict, tag: str = "Self-Grooming"):
    """
    Get frame ranges from annotation file.
    It returns a list of [start, end] entries.
    """
    frame_ranges = []
    for ann in ann_file["tags"]:
        if ann["name"] == tag:
            frame_ranges.append(ann["frameRange"])
    return frame_ranges

def split_range(start: int, end: int, fps: float, total_frames: int, max_clip_duration: float = 5) -> list:
    """
    Returns a list of segments for the given range.
      - If the range has exactly one frame, pad the clip to one second.
      - If the range spans more than `max_clip_duration` seconds, split into equal segments.
    Each segment is a tuple (seg_start, seg_end) in frame numbers.
    """
    segments = []
    clip_frames = end - start + 1

    # Pad the clip to one second if it has only one frame.
    fps = int(round(fps))
    if clip_frames == 1:
        half = fps // 2
        start = max(0, start - half)
        end = min(total_frames - 1, start + fps - 1)
        clip_frames = end - start + 1

    # If the clip is longer than max_clip_duration sec then split it.
    ten_sec_frames = int(max_clip_duration * fps)
    if clip_frames > ten_sec_frames:
        seg_count = math.ceil(clip_frames / ten_sec_frames)
        base_seg_length = clip_frames // seg_count
        extra = clip_frames % seg_count
        current_start = start
        for i in range(seg_count):
            # Distribute extra frame to the first 'extra' segments.
            current_seg_length = base_seg_length + (1 if i < extra else 0)
            current_end = current_start + current_seg_length - 1
            segments.append((current_start, current_end))
            current_start = current_end + 1
    else:
        segments.append((start, end))

    return segments

def main(video_file: str, ann_file: str, output_dir: str, info_file: str):
    video_path = Path(video_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open video with decord to fetch FPS and total frames.
    vr = decord.VideoReader(str(video_path))
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")

    # Load annotation file and get frame ranges.
    with open(ann_file, "r") as f:
        ann = json.load(f)
    ranges = get_frame_ranges(ann)

    info_lines = []
    clip_counter = 1

    # Process each annotated frame range.
    for frame_range in tqdm(ranges):
        start, end = frame_range
        segments = split_range(start, end, fps, total_frames)

        for seg in segments:
            seg_start, seg_end = seg
            # Calculate start time and duration in seconds.
            seg_start_time = seg_start / fps
            seg_duration = (seg_end - seg_start + 1) / fps

            clip_name = f"clip_{clip_counter:03d}.mp4"
            output_clip = out_dir / clip_name

            # Build and run ffmpeg command.
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if exists.
                "-i", str(video_path),
                "-ss", str(seg_start_time),
                "-t", str(seg_duration),
                # "-vf", "scale=224:224",  # Width:Height
                "-c:v", "libx264",  # Use H.264 codec
                "-preset", "medium",  # Balance between speed and quality
                "-crf", "18",      # Quality setting (lower = better quality)
                "-pix_fmt", "yuv420p",
                str(output_clip)
            ]
            # Using subprocess; output is suppressed.
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Add info line: clip file name, start frame, end frame.
            info_lines.append(f"{clip_name} {seg_start} {seg_end}")
            clip_counter += 1

    # Write all info lines to the info file.
    with open(info_file, "w") as f:
        for line in info_lines:
            f.write(line + "\n")

if __name__ == "__main__":

    video_file = "HOM Mice F.2632_HOM 12 Days post tre/12 Days post tre/video_320px/GL020560.mp4"
    clips_dir = "clips_positive_320px"
    info_file = "info_positive.txt"
    ann_file = "HOM Mice F.2632_HOM 12 Days post tre/12 Days post tre/ann/GL020560.MP4.json"

    video_file = Path(video_file)
    # auto:
    # ann_file = video_file.parent.parent / f"ann/{video_file.stem+video_file.suffix}.json"
    ann_file = Path(ann_file)
    output_dir = video_file.parent.parent / f"{video_file.stem}-{clips_dir}"
    main(video_file, ann_file, output_dir, output_dir.parent / info_file)