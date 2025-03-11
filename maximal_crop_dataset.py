from argparse import Namespace
import json
import os
from pathlib import Path
import numpy as np
import torch
import decord
from PIL import Image
from torchvision import transforms
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from kinetics import VideoClsDataset


class MaximalCropDataset(VideoClsDataset):

    def __init__(self, anno_path, data_path, det_anno_path, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3, args=None,
                 ):
        assert mode != "test"
        if not args:
            args = Namespace()
            args.num_sample = 2
            args.reprob = 0.
            args.train_interpolation = 'bicubic'
            args.aa = 'rand-m7-n4-mstd0.5-inc1'
            args.data_set = 'Kinetics-400'
        super(MaximalCropDataset, self).__init__(
            anno_path, data_path, mode, clip_len, frame_sample_rate, crop_size, short_side_size,
            new_height, new_width, keep_aspect_ratio, num_segment, num_crop, test_num_segment, test_num_crop, args)
        
        self.det_bbox_padding = 0.1
        self._scl, self._asp = (
            [0.7, 1.0],
            [0.75, 1.3333],
        )
        self.translate_fraction = 0.10

        self.det_anno_path = det_anno_path
        
    def loadvideo_decord(self, sample, sample_rate_scale=1):
        video_path = sample
        # T H W C
        buffer = super().loadvideo_decord(video_path, sample_rate_scale)
        all_index = self._sample_indexes(video_path, sample_rate_scale)
        min_frame_idx = min(all_index)
        max_frame_idx = max(all_index)
        ann_path = self._get_det_ann_path(video_path)
        x1, y1, x2, y2 = self._get_crop_bbox(ann_path, min_frame_idx, max_frame_idx)
        buffer = buffer[:, y1:y2, x1:x2, :]
        return buffer
    
    def _get_det_ann_path(self, video_path):
        ann_path = "/".join(video_path.split("/")[-2:])
        ann_path = f"{self.det_anno_path}/{ann_path}.json"
        return ann_path
    
    def _sample_indexes(self, video_path, sample_rate_scale=1):
        vr = VideoReader(video_path)
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = len(vr) // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate((index, np.ones(self.clip_len - seg_len // self.frame_sample_rate) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i*seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        return all_index

    def _get_crop_bbox(self, ann_path, min_frame_idx, max_frame_idx):
        with open(ann_path, 'r') as f:
            ann = json.load(f)
        figures = [fig for frame in ann['frames'][min_frame_idx:max_frame_idx+1] for fig in frame['figures']]
        x1, y1, x2, y2 = get_maximal_bbox(figures)
        x1, y1, x2, y2 = get_square_bbox((x1, y1, x2, y2), padding=self.det_bbox_padding)
        w, h = ann["size"]["width"], ann["size"]["height"]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        return x1, y1, x2, y2


def get_maximal_bbox(figures):
    """
    Get the maximal bounding box that encompasses all other boxes.
    Returns a bbox with min x1, min y1, max x2, max y2 from all bboxes.
    """
    if not figures:
        return None
    
    # Initialize with the first bbox
    first_bbox = figures[0]['geometry']['points']['exterior']
    min_x = first_bbox[0][0]
    min_y = first_bbox[0][1]
    max_x = first_bbox[1][0]
    max_y = first_bbox[1][1]
    
    # Find the min/max coordinates across all bboxes
    for figure in figures:
        bbox = figure['geometry']['points']['exterior']
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]
        
        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)
    
    return min_x, min_y, max_x, max_y


def get_square_bbox(bbox, padding=0.0):
    """
    Convert rectangular bbox to square by adding padding to the shortest edge.
    Adds additional padding to all sides.
    Returns (x, y, x2, y2) where (x, y) is the top-left corner and (x2, y2) is the bottom-right corner.
    """
    x1, y1, x2, y2 = bbox
    
    width = x2 - x1
    height = y2 - y1
    
    if width > height:
        # Add padding to height
        diff = width - height
        y1 = max(0, y1 - diff // 2)
        y2 = y1 + width
    else:
        # Add padding to width
        diff = height - width
        x1 = max(0, x1 - diff // 2)
        x2 = x1 + height
    
    # Add additional padding
    size = x2 - x1  # At this point, size = width = height (it's square)
    pad_px = int(padding * size)
    x1 = max(0, x1 - pad_px)
    y1 = max(0, y1 - pad_px)
    x2 = x2 + pad_px
    y2 = y2 + pad_px
    
    return int(x1), int(y1), int(x2), int(y2)
