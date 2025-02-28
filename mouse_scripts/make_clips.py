import json
import math
from pathlib import Path
import random
import video_utils as V
from tqdm import tqdm
import pandas as pd

def get_frame_ranges(ann_file: dict, tag: str):
    """
    Get frame ranges from annotation file.
    It returns a list of [start, end] entries.
    """
    frame_ranges = []
    for ann in ann_file["tags"]:
        if ann["name"] == tag:
            frame_ranges.append(ann["frameRange"])
    return frame_ranges

def merge_overlapping_ranges(ranges: list) -> list:
    ranges = sorted(ranges, key=lambda x: x[0])
    merged = []
    for rng in ranges:
        if not merged or rng[0] > merged[-1][1] + 1:
            merged.append(rng)
        else:
            merged[-1][1] = max(merged[-1][1], rng[1])
    ranges = merged
    return ranges

def filter_ranges_outside_video(ranges: list, total_frames: int) -> list:
    return [[start, end] for start, end in ranges if 0 <= start < total_frames and 0 <= end < total_frames]

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

def make_pos_clips_for_tag(
    video_file: str,
    ann_file: str,
    output_dir: str,
    target_short_edge: int,
    tag: str,
    label: int,
):
    video_path = Path(video_file)
    video_name = video_path.stem
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = V.get_fps(video_path)
    total_frames = V.get_total_frames(video_path)

    # Load annotation file and get frame ranges.
    with open(ann_file, "r") as f:
        ann = json.load(f)
    
    ranges = get_frame_ranges(ann, tag=tag)
    ranges = filter_ranges_outside_video(ranges, total_frames)
    ranges = merge_overlapping_ranges(ranges)

    info = []
    clip_counter = 1
    # Process each annotated frame range.
    for frame_range in ranges:
        start, end = frame_range
        segments = split_range(start, end, fps, total_frames)

        for seg in segments:
            seg_start, seg_end = seg
            clip_name = f"{video_name}_clip_{clip_counter:03d}.mp4"
            tag_name = tag.replace("/", "-").replace(" ", "_")
            output_clip = out_dir / tag_name / clip_name
            output_clip.parent.mkdir(parents=True, exist_ok=True)
            width, height = V.get_video_dimensions(video_path)
            new_width, new_height = V.calculate_resize(width, height, target_short_edge=target_short_edge)
            V.extract_clip(video_path, seg_start, seg_end, new_width, new_height, fps, output_clip)
            
            # orig_file, clip_file, start, end, label(1,2)
            info.append([str(video_path), str(output_clip), seg_start, seg_end, label])
            clip_counter += 1

    return info

def make_positives(input_dir: str, output_dir: str, min_size):
    p = Path(input_dir)
    paths = list(p.rglob("*.MP4"))
    paths += list(p.rglob("*.mp4"))
    print(f"Found {len(paths)} video files.")

    # find duplicates
    paths = unique_video_names(paths)

    LABELS = {"Self-Grooming": 1, "Head/Body TWITCH": 2}

    infos = []
    for i, video_file in enumerate(tqdm(paths)):
        ann_file = video_file.parent.parent / f"ann/{video_file.name}.json"
        assert ann_file.exists(), f"Annotation file not found: {ann_file}"
        for tag, label in LABELS.items():
            infos += make_pos_clips_for_tag(video_file, ann_file, output_dir, min_size, tag, label)
            # validate_decord(infos[-1][1])
        print(f"{i+1}/{len(paths)}")

    return infos

def make_neg_clips_for_tag(
    video_file: str, 
    output_dir: str, 
    target_short_edge: int,
    target_length: int,
    skip_ranges: list,
    tag: str = "idle",
    label: int = 0,
    min_clip_duration: int = 3,
    max_clip_duration: int = 5,
):
    video_path = Path(video_file)
    video_name = video_path.stem
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = V.get_fps(video_path)
    total_frames = V.get_total_frames(video_path)

    # Merge overlapping skip ranges.
    skip_ranges = merge_overlapping_ranges(skip_ranges)

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
    info = []

    # Process non-skip intervals and extract random clips.
    for interval in non_skip_intervals:
        interval_start, interval_end = interval
        t = interval_start
        while t + clip_min_frames - 1 <= interval_end and cumulative_clip_frames < target_length:
            available = interval_end - t + 1
            if available < clip_min_frames:
                break
            clip_length = random.randint(clip_min_frames, min(clip_max_frames, available))
            start_frame = t
            end_frame = t + clip_length - 1
            clip_name = f"{video_name}_clip_{clip_counter:03d}.mp4"
            output_clip = out_dir / tag / clip_name
            output_clip.parent.mkdir(parents=True, exist_ok=True)
            width, height = V.get_video_dimensions(video_path)
            new_width, new_height = V.calculate_resize(width, height, target_short_edge=target_short_edge)
            V.extract_clip(video_path, start_frame, end_frame, new_width, new_height, fps, output_clip)
            
            # orig_file, clip_file, start, end, label=0
            info.append([str(video_path), str(output_clip), start_frame, end_frame, label])

            cumulative_clip_frames += clip_length
            clip_counter += 1
            t = end_frame + 1
            if cumulative_clip_frames >= target_length:
                break

    return info

def make_negatives(pos_df: pd.DataFrame, output_dir: str, min_size, target_length):
    grouped = pos_df.groupby("orig_file")
    infos = []
    for i, (video_file, group_df) in enumerate(tqdm(grouped)):
        skip_ranges = group_df[["start", "end"]].values.tolist()
        infos += make_neg_clips_for_tag(
            video_file, output_dir, min_size, target_length=target_length, skip_ranges=skip_ranges
        )
        print(f"{i+1}/{len(grouped)}")

    return infos

def unique_video_names(paths: list):
    """Return list of paths with unique video names (stems).
    If duplicates are found, keep only the first occurrence.
    
    Args:
        paths: List of Path objects pointing to video files
    Returns:
        List of Path objects with unique stems
    """
    seen = set()
    unique_paths = []
    for path in paths:
        if path.stem not in seen:
            seen.add(path.stem)
            unique_paths.append(path)
    if len(unique_paths) < len(paths):
        print(f"Found {len(paths) - len(unique_paths)} duplicate video names in the input list.")
    return unique_paths

def validate_decord(video_file: str):
    import decord
    import numpy as np
    vr = decord.VideoReader(str(video_file))
    vr.seek(0)
    n = len(vr)
    idxs = np.random.randint(0, n, 10).tolist()
    frames = vr.get_batch(idxs).asnumpy()
    return frames

if __name__ == "__main__":
    input_dir = "MP_TRAIN_3"
    output_dir = "output"
    min_size = 480

    Path(output_dir).mkdir(parents=True, exist_ok=False)

    pos_infos = make_positives(input_dir=input_dir, output_dir=output_dir, min_size=min_size)
    df = pd.DataFrame(pos_infos, columns=["orig_file", "clip_file", "start", "end", "label"])
    df.to_csv("positives.csv", index=False)
    print(f"Saved {len(pos_infos)} positive clips to 'positives.csv'")
    
    # Calculate average frame range length per video file
    df['range_length'] = df['end'] - df['start'] + 1  # +1 because end frame is inclusive
    avg_lengths = df.groupby('orig_file')['range_length'].agg(['sum', 'count', 'mean'])
    avg_lengths.columns = ['total_frames', 'clip_count', 'avg_length_per_clip']
    avg_lengths.to_csv("avg_lengths_positives.csv")

    target_length = int(avg_lengths['total_frames'].mean())
    print(f"Average target length: {target_length}")

    neg_infos = make_negatives(pos_df=df, output_dir=output_dir, min_size=min_size, target_length=target_length)
    df = pd.DataFrame(neg_infos, columns=["orig_file", "clip_file", "start", "end", "label"])
    df.to_csv("negatives.csv", index=False)
    print(f"Saved {len(neg_infos)} negative clips to 'negatives.csv")

    # concatenate positive and negative clips
    df = pd.concat([pd.read_csv("positives.csv"), pd.read_csv("negatives.csv")])
    df.to_csv("clips.csv", index=False)
    print(f"Saved {len(df)} clips to 'clips.csv'")
    print("Done.")
