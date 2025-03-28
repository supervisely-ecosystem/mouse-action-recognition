import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

from timm.models import create_model
from tqdm import tqdm

from src.inference.video_sliding_window import VideoSlidingWindow
from src.inference.maximal_bbox_sliding_window import MaximalBBoxSlidingWindow
import utils
from src.inference.arg_parser import get_parser
from src.bbox_utils import get_maximal_bbox
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


def load_mvd(checkpoint):
    checkpoint = str(checkpoint)
    experiment_name = checkpoint.split('/')[-3]
    print(f"Experiment name: {experiment_name}")
    checkpoint = Path(checkpoint)
    assert checkpoint.exists(), f"Checkpoint {checkpoint} does not exist."
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
    return model, opts


def load_detector(session_url="http://supervisely-utils-rtdetrv2-inference-1:8000"):
    import supervisely as sly
    from supervisely.nn.inference import SessionJSON
    api = sly.Api()
    detector = SessionJSON(api, session_url=session_url)
    return detector


def predict_video_with_detector(video_path, model, detector, opts, stride, pbar=None):

    # Read the video
    dataset = MaximalBBoxSlidingWindow(
        video_path,
        detector=detector,
        num_frames=opts.num_frames,
        frame_sample_rate=opts.sampling_rate,
        input_size=opts.input_size,
        stride=stride,
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
    if pbar is None:
        iterator = tqdm(data_loader)
    else:
        iterator = data_loader
    for input, frame_indices, bboxes in iterator:
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
            frame_range = [int(frame_idxs[0]), int(frame_idxs[-1]) + opts.sampling_rate]  # inclusive range
            bbox = bboxes[i]
            predictions.append({
                'frame_range': frame_range,
                'label': predicted_class,
                'confidence': confidence,
                'probabilities': prob.tolist(),
                'maximal_bbox': bbox,
            })
        if pbar is not None:
            pbar.update()

    return predictions

def merge_predictions(predictions: list):
    """
    Merge predictions based on frame ranges and labels.
    """
    predictions.sort(key=lambda x: x['frame_range'][0])
    merged_predictions = []
    for pred in predictions:
        start_frame, end_frame = pred['frame_range']
        label = pred['label']
        confidence = pred['confidence']
        bbox = pred.get('maximal_bbox')
        
        # If this is the first segment or it doesn't overlap with previous segment
        if not merged_predictions or start_frame > merged_predictions[-1]['frame_range'][1] or label != merged_predictions[-1]['label']:
            merged_predictions.append({
                'frame_range': [start_frame, end_frame],
                'label': label,
                'confidence': confidence,
                'bbox': bbox,
            })
        else:
            # Extend the previous segment
            merged_predictions[-1]['frame_range'][1] = max(merged_predictions[-1]['frame_range'][1], end_frame)
            # Update confidence to the max of the two segments
            merged_predictions[-1]['confidence'] = max(merged_predictions[-1]['confidence'], confidence)
            # Update bbox if needed
            if bbox is not None:
                merged_predictions[-1]['bbox'] = get_maximal_bbox([merged_predictions[-1]['bbox'], bbox])
    
    print(f"Merged {len(predictions)} predictions into {len(merged_predictions)}")
    return merged_predictions


def extract_positive_predictions(predictions: list):
    """
    Extract positive predictions (label != 0) from the list of predictions.
    """
    positive_predictions = [pred for pred in predictions if pred['label'] != 0]
    print(f"Found {len(positive_predictions)} positive predictions out of {len(predictions)} total")
    return positive_predictions


def threshold_predictions(predictions: list, conf: float = 0.5):
    """
    Filter predictions based on confidence threshold.
    """
    filtered_predictions = [pred for pred in predictions if pred['confidence'] >= conf]
    print(f"Filtered {len(predictions) - len(filtered_predictions)} predictions below confidence {conf}")
    return filtered_predictions


def postprocess_predictions(predictions: list, conf=None):
    predictions = merge_predictions(predictions)
    predictions = extract_positive_predictions(predictions)
    if conf is not None:
        predictions = threshold_predictions(predictions, conf)
    return predictions
