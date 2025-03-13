import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import json
import os
from pathlib import Path
import decord
import numpy as np
from matplotlib import pyplot as plt

from timm.models import create_model
from tqdm import tqdm

from datasets import build_dataset
from engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test, merge
from video_sliding_window import VideoSlidingWindow
from maximal_bbox_sliding_window import MaximalBBoxSlidingWindow
import utils
from arg_parser import get_parser
from inference_visualizations import draw_timeline
import modeling_finetune  # register the new model
import supervisely as sly
from supervisely.nn.inference import SessionJSON


def build_model(args):
    device = torch.device(args.device)
    cudnn.benchmark = True
    model = create_model(
        args.model,
        pretrained=False,
        img_size=args.input_size,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_cls_token=args.use_cls_token,
        fc_drop_rate=args.fc_drop_rate,
        use_checkpoint=args.use_checkpoint,
    )

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    print("Loading ckpt from %s" % args.finetune)
    checkpoint = torch.load(args.finetune, map_location='cpu')
    checkpoint_model = None
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters // 1e6, 'M')

    return model


def parse_config(config_text):
    config_dict = {}
    
    # Split by lines and process each line
    for line in config_text.strip().split('\n'):
        # Skip empty lines
        if not line.strip():
            print("Skipping empty line")
            continue
            
        # Split by whitespace and take first and last elements
        parts = line.split(None, 1)
        if len(parts) == 2:
            key = parts[0]
            value = parts[1].strip()
            
            # Convert values to appropriate types
            if value.lower() == 'false':
                value = False
            elif value.lower() == 'true':
                value = True
            elif value.replace('.', '').isdigit():  # Check if number (including decimals)
                value = float(value) if '.' in value else int(value)
            elif value.lower() == 'none':
                continue
                
            config_dict[key] = value
        else:
            print(f"Skipping line: {line}")
    
    return config_dict

if __name__ == '__main__':
    checkpoint = "/root/volume/OUTPUT/MP_TRAIN_3_maximal_crop_2025-03-11_15-09-26/checkpoint-best/mp_rank_00_model_states.pt"
    video_path = "/root/volume/data/mouse/HOM Mice F.2632_HOM 12 Days post tre/12 Days post tre/video/GL010560.MP4"
    ann_path = Path(video_path).parent.parent / "ann" / (Path(video_path).name + ".json")
    STRIDE = 8  # 8x2=16 of 32
    detector_url = "http://supervisely-utils-rtdetrv2-inference-1:8000"
    api = sly.Api()
    detector = SessionJSON(api, session_url=detector_url)

    experiment_name = checkpoint.split('/')[-3]
    print(f"Experiment name: {experiment_name}")
    checkpoint = Path(checkpoint)
    assert checkpoint.exists(), f"Checkpoint {checkpoint} does not exist."
    assert Path(video_path).exists(), f"Video {video_path} does not exist."
    assert ann_path.exists(), f"Annotation {ann_path} does not exist."
    output_dir = checkpoint.parent.parent
    
    config_file = output_dir / "config.txt"
    with open(config_file, 'r') as f:
        config_text = f.read()
    config = parse_config(config_text)

    parser = get_parser()
    # Get the allowed arguments from parser
    allowed_args = {action.dest for action in parser._actions}
    
    # Filter config_dict to only include allowed arguments
    valid_config = {k: v for k, v in config.items() if k in allowed_args}
    
    # Convert dict to list of args and parse
    args_list = []
    for key, value in valid_config.items():
        if value is True or value is False:
            continue
        args_list.extend([f'--{key}', str(value)])
    opts = parser.parse_args(args_list)
    opts.finetune = checkpoint
    opts.eval = True
    print(opts)

    # Build the model
    model = build_model(opts)
    model.eval()
    
    # Read the video
    dataset = MaximalBBoxSlidingWindow(
        video_path,
        detector=detector,
        num_frames=opts.num_frames,
        frame_sample_rate=opts.sampling_rate,
        input_size=opts.input_size,
        stride=STRIDE,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=MaximalBBoxSlidingWindow.collate_fn,
    )

    # Inference
    device = opts.device
    print(f"Inference on {video_path}")
    print(f"dataset length: {len(dataset)}")

    predictions = []
    for input, frame_indices in tqdm(data_loader):
        input = input.to(device)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output = model(input)
        
        # Get probabilities from the model output
        probs = torch.softmax(output, dim=1)
        for i, frame_idxs in enumerate(frame_indices):
            prob = probs[i].cpu().numpy()
            predicted_class = int(np.argmax(prob))
            confidence = float(prob[predicted_class])
            frame_range = [int(frame_idxs[0]), int(frame_idxs[-1])]
            predictions.append({
                'frame_range': frame_range,
                'label': predicted_class,
                'confidence': confidence,
                'probabilities': prob.tolist(),
            })

    # Save predictions to JSON file
    os.makedirs("results", exist_ok=True)
    output_json_path = f"results/predictions_{experiment_name}.json"
    with open(output_json_path, 'w') as f:
        json.dump(predictions, f, indent=4)
        
    class_names = ["idle", "Self-Grooming", "Head/Body Twitch"]
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()    
    draw_timeline(predictions, fps, experiment_name=experiment_name, class_names=class_names,
                 figsize=(15, 7))

    # Display additional statistics
    # Extract probabilities from the predictions list
    all_probs = np.array([pred['probabilities'] for pred in predictions])
    num_windows = len(predictions)
    
    # Get the most probable class for each window
    dominant_classes = np.argmax(all_probs, axis=1)
    class_counts = np.bincount(dominant_classes)
    top_classes = np.argsort(-class_counts)[:5]
    print("\nDominant classes in the video:")
    for cls in top_classes:
        if class_counts[cls] > 0:
            percentage = (class_counts[cls] / num_windows) * 100
            print(f"Class {cls}: {percentage:.2f}% ({class_counts[cls]} windows)")
