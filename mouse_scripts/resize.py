import os
import subprocess
from pathlib import Path
from tqdm import tqdm

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

def resize_video(input_path, output_path, target_short_edge=320):
    """Resize video maintaining aspect ratio with short edge = target_short_edge"""
    # make dirs
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get original dimensions
    width, height = get_video_dimensions(input_path)
    
    # Calculate new dimensions and ensure they're even
    if width < height:
        new_width = target_short_edge
        new_height = int(height * (target_short_edge / width))
    else:
        new_height = target_short_edge
        new_width = int(width * (target_short_edge / height))
        
    # Make dimensions even
    new_width = new_width + (new_width % 2)
    new_height = new_height + (new_height % 2)
    
    # Construct FFmpeg command
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-vf', f'scale={new_width}:{new_height}',
        '-c:v', 'libx264',  # Use H.264 codec
        '-crf', '23',       # Constant rate factor (18-28 is good, lower = better quality)
        '-c:a', 'copy',     # Copy audio without re-encoding
        '-y',               # Overwrite output file if it exists
        str(output_path)
    ]
    
    # Execute FFmpeg command with suppressed output
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,  # Capture stderr for error reporting
            encoding='utf-8'
        )
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")

def process_videos(input_dir, output_dir):
    """Process all videos in input directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each video file
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    files = list(input_path.glob('**/*'))[:100]
    for video_file in tqdm(files):
        if video_file.suffix.lower() in video_extensions:
            # Create relative output path
            rel_path = video_file.relative_to(input_path)
            output_file = output_path / rel_path
            
            # Create parent directories if they don't exist
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # print(f"Processing: {video_file}")
            try:
                resize_video(video_file, output_file)
                # print(f"Successfully processed: {output_file}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {video_file}: {e}")

if __name__ == "__main__":
    # Example usage
    # input_directory = "k400/val"
    # output_directory = "k400/val_resized"
    
    # process_videos(input_directory, output_directory)

    folder_suffix = "224px"
    video_file = "HOM Mice F.2632_HOM 12 Days post tre/12 Days post tre/video/GL020560.MP4"
    p = Path(video_file)
    output_file = p.parent.parent / f"video_{folder_suffix}/{p.name}"
    print(output_file)
    resize_video(video_file, output_file)