import os

# cmd = 'ffmpeg', '-y', '-ss', '62762.199499999995', '-i', 'MP_TRAIN_3/HOM Mice/datasets/F.2823_HOM/datasets/28 Days post tre/video/GL010660.MP4', '-t', '-62053.4915', '-vf', 'scale=848:480', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', '-force_key_frames', 'expr:gte(t,0)', '-an', 'output/Self-Grooming/GL010660_clip_104.mp4'

cmd = [
    'ffmpeg',
    '-y',
    '-ss', '62762.199499999995',
    '-i', 'MP_TRAIN_3/HOM Mice/datasets/F.2823_HOM/datasets/28 Days post tre/video/GL010660.MP4',
    '-t', '-62053.4915',
    '-vf', 'scale=848:480',
    '-c:v', 'libx264',
    '-preset', 'fast',
    '-crf', '20',
    '-pix_fmt', 'yuv420p',
    '-force_key_frames', 'expr:gte(t,0)',
    '-an',
    'tmp.mp4'
]

from make_clips import make_pos_clips_for_tag


tag = "Self-Grooming"
label = 1
video_file = "MP_TRAIN_3/HOM Mice/datasets/F.2823_HOM/datasets/28 Days post tre/video/GL010660.MP4"
ann_file = "MP_TRAIN_3/HOM Mice/datasets/F.2823_HOM/datasets/28 Days post tre/ann/GL010660.MP4.json"
output_dir = "output"
make_pos_clips_for_tag(video_file, ann_file, output_dir, 480, tag, label)