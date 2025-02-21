from pathlib import Path
import shutil

def get_clips(directory: Path):
    """Get all MP4 files from directory."""
    return sorted([f"{f.parent.name}/{f.name}" for f in directory.glob("*.mp4")])

def main(pos_dir: Path, neg_dir: Path, output_file: Path):
    # Get clips from both directories
    pos_clips = get_clips(pos_dir)
    neg_clips = get_clips(neg_dir)

    # Create annotation lines
    annotations = []
    # Add positive clips (label 1)
    for clip in pos_clips:
        annotations.append(f"{clip} 1")
    # Add negative clips (label 0)
    for clip in neg_clips:
        annotations.append(f"{clip} 0")

    # Write annotations to file
    with open(output_file, "w") as f:
        f.write("\n".join(annotations))
    
    print(f"Created annotation file with {len(annotations)} entries")
    print(f"Positive clips: {len(pos_clips)}")
    print(f"Negative clips: {len(neg_clips)}")

if __name__ == "__main__":
    base_dir = Path("HOM Mice F.2632_HOM 12 Days post tre/12 Days post tre")
    pos_dir = base_dir / "GL020560-clips_positive_320px"
    neg_dir = base_dir / "GL020560-clips_negative_320px"
    output_file = base_dir / "annotations.txt"
    
    main(pos_dir, neg_dir, output_file)
    shutil.copy(output_file, base_dir / "train.csv")
    shutil.copy(output_file, base_dir / "val.csv")
