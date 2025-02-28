import subprocess

def get_video_dimensions(video_path):
    """Get video dimensions using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0',
        str(video_path)
    ]
    output = subprocess.check_output(cmd).decode('utf-8').strip().split(',')
    return int(output[0]), int(output[1])

def get_fps(video_path):
    """Get video FPS using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'csv=p=0',
        str(video_path)
    ]
    output = subprocess.check_output(cmd).decode('utf-8').strip()
    if '/' in output:
        numerator, denominator = output.split('/')
        return int(numerator) / int(denominator)
    else:
        return float(output)

def get_total_frames(video_path):
    """Get total frames using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames',
        '-of', 'csv=p=0',
        str(video_path)
    ]
    output = subprocess.check_output(cmd).decode('utf-8').strip()
    
    # Some videos might not have frame count metadata
    if output and output != 'N/A':
        return int(output)
    else:
        # Fallback: count frames using fps and duration
        print("Warning: Frame count not found in metadata. Counting frames manually.")
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=duration',
            '-of', 'csv=p=0',
            str(video_path)
        ]
        duration = float(subprocess.check_output(cmd).decode('utf-8').strip())
        fps = get_fps(video_path)
        return int(duration * fps)

def calculate_resize(original_width, original_height, target_short_edge=320):
    """Calculate new dimensions maintaining aspect ratio with short edge = target_short_edge"""
    # Calculate new dimensions and ensure they're even
    if original_width < original_height:
        new_width = target_short_edge
        new_height = int(original_height * (target_short_edge / original_width))
    else:
        new_height = target_short_edge
        new_width = int(original_width * (target_short_edge / original_height))
        
    # Make dimensions even
    new_width = new_width + (new_width % 2)
    new_height = new_height + (new_height % 2)

    return new_width, new_height

def extract_clip(video_path, start, end, width, height, fps, output_clip):
    """Extract a clip from a video using ffmpeg"""
    start_time = start / fps
    duration = (end - start + 1) / fps
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if exists.
        "-ss", str(start_time),
        "-i", str(video_path),
        "-t", str(duration),
        "-vf", f"scale={width}:{height}",
        "-c:v", "libx264",  # Use H.264 codec
        "-preset", "fast",  # Balance between speed and quality
        "-crf", "20",  # Quality setting (lower = better quality)
        "-pix_fmt", "yuv420p",
        "-force_key_frames", "expr:gte(t,0)",  # Force keyframe at start
        "-an",  # Remove audio to speed up
        str(output_clip)
    ]
    # Using subprocess; output is suppressed.
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)