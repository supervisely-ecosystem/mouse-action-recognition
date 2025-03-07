import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import json
import os
from pathlib import Path

from timm.models import create_model
from tqdm import tqdm

from datasets import build_dataset
from engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test, merge
from video_sliding_window import VideoSlidingWindow
import utils
import modeling_finetune  # register the new model


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


def get_parser():
    parser = argparse.ArgumentParser('MVD fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default= 2)
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')

    parser.add_argument('--fc_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)

    parser.add_argument('--remove_pos_emb', action='store_true', default=False)

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    parser.add_argument('--use_cls_token', action='store_true', default=False)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    # parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
    #                     help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--num_sample', type=int, default=2,
                        help='Repeated_aug (default: 2)')
    parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=5)
    parser.add_argument('--test_num_crop', type=int, default=3)
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters
    parser.add_argument('--data_root', default=None, type=str,
                        help='dataset path root')
    parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str,
                        help='path of dataset file list')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=400, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--data_set', default='Kinetics-400', choices=['Kinetics-400', 'SSV2', 'UCF101', 'HMDB51','image_folder'],
                        type=str, help='dataset')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--resume_best', action='store_true', default=False)
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)
    parser.add_argument('--no_save_best_ckpt', action='store_true', default=False)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--merge_test', action='store_true', default=False)
    parser.add_argument('--eval_log_name', default='log_eval', type=str,
                        help='Perform evaluation only, the log name')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)

    return parser


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
    checkpoint = "/root/volume/MP_TRAIN_3_2025-03-01_11-18-45/checkpoint-9/mp_rank_00_model_states.pt"
    video_path = "/root/volume/data/mouse/HOM Mice F.2632_HOM 12 Days post tre/12 Days post tre/video/GL010560.MP4"
    checkpoint = Path(checkpoint)
    assert checkpoint.exists(), f"Checkpoint {checkpoint} does not exist."
    assert Path(video_path).exists(), f"Video {video_path} does not exist."
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
    
    # Read the video
    dataset = VideoSlidingWindow(
        video_path,
        num_frames=opts.num_frames,
        frame_sample_rate=opts.sampling_rate,
        input_size=opts.input_size,
        stride=5,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    # Perform evaluation
    device = opts.device
    num_frames = opts.num_frames
    frame_sample_rate = opts.sampling_rate
    stride = 5
    print(f"Performing evaluation on {video_path}")
    print(len(dataset))
    all_probs = []
    for videos in tqdm(data_loader):
        videos = videos.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast() and torch.no_grad():
            output = model(videos)

        prob = torch.softmax(output, dim=1)
        all_probs.append(prob)

    all_probs = torch.cat(all_probs, dim=0).cpu().numpy()  # (num_windows, num_classes)
    print(all_probs.shape)

    from matplotlib import pyplot as plt
    import numpy as np

    num_windows = all_probs.shape[0]
    num_classes = all_probs.shape[1]

    # Calculate frame timestamps for each window
    window_centers = [(num_frames * frame_sample_rate) // 2 + i * stride for i in range(num_windows)]
    
    # Convert frame indices to seconds (assuming 30fps, adjust if different)
    fps = dataset.vr.get_avg_fps()
    time_in_seconds = [frame / fps for frame in window_centers]
        
    # Create the visualization
    plt.figure(figsize=(15, 7))
    
    for idx in range(num_classes):
        plt.plot(time_in_seconds, all_probs[:, idx], linewidth=2, label=f"Class {idx}")
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Probability')
    plt.title('Class Probabilities Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the visualization
    vis_path = os.path.join('timeline_visualization.png')
    plt.savefig(vis_path)
    print(f"Visualization saved to {vis_path}")
    
    # Also save the raw probabilities for further analysis
    np.save(os.path.join('class_probabilities.npy'), all_probs)
    
    # Display additional statistics
    # Get the most probable class for each window
    dominant_classes = np.argmax(all_probs, axis=1)
    class_counts = np.bincount(dominant_classes)
    top_classes = np.argsort(-class_counts)[:5]
    print("\nDominant classes in the video:")
    for cls in top_classes:
        if class_counts[cls] > 0:
            percentage = (class_counts[cls] / num_windows) * 100
            print(f"Class {cls}: {percentage:.2f}% ({class_counts[cls]} windows)")


