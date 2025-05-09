from src.inference.arg_parser import get_parser
from timm.models import create_model
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import utils
from pathlib import Path
from supervisely import logger

def load_mvd(checkpoint_path, config_path):
    logger.info(f"Loading MVD from {checkpoint_path} and {config_path}")
    with open(config_path, 'r') as f:
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
    opts.finetune = checkpoint_path
    opts.eval = True

    logger.info(f"Building model from config")
    model = build_model(opts)
    model.eval()
    return model, opts

def parse_config(config_text):
    logger.info(f"Parsing config")
    config_dict = {}
    for line in config_text.strip().split('\n'):
        if not line.strip():
            logger.debug("Skipping empty line")
            continue

        parts = line.split(None, 1)
        if len(parts) == 2:
            key = parts[0]
            value = parts[1].strip()
            
            if value.lower() == 'false':
                value = False
            elif value.lower() == 'true':
                value = True
            elif value.replace('.', '').isdigit():
                value = float(value) if '.' in value else int(value)
            elif value.lower() == 'none':
                continue

            config_dict[key] = value
        else:
            logger.debug(f"Skipping line: {line}")

    logger.debug(f"Parsed config: {config_dict}")
    return config_dict

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
    logger.info(f"Patch size = {str(patch_size)}")
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    logger.info(f"Loading ckpt from {args.finetune}")
    checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
    checkpoint_model = None
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            logger.debug(f"Load state_dict by model_key = {model_key}")
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters // 1e6} M')
    return model