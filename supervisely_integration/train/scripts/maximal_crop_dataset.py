import os
from pathlib import Path

from maximal_crop_dataset import MaximalCropDataset


class MaximalCropDatasetTrainApp(MaximalCropDataset):

    def _get_det_ann_path(self, video_path):
        ds_path = Path(video_path).parent.parent
        video_name = Path(video_path).name
        ann_path = os.path.join(ds_path, "ann", f"{video_name}.json")
        return ann_path
