import torch
from video_sliding_window import VideoSlidingWindow
from maximal_crop_dataset import get_maximal_bbox, get_square_bbox
from supervisely.nn.inference import SessionJSON
from tempfile import TemporaryDirectory
from PIL import Image
from collections import OrderedDict
import os

class MaximalBBoxSlidingWindow(VideoSlidingWindow):
    def __init__(self, video_path, detector, num_frames=16,
                 frame_sample_rate=2, input_size=224,
                 stride=5, bbox_padding=0.05):
        super().__init__(video_path, num_frames, frame_sample_rate, input_size, stride)
        assert isinstance(detector, SessionJSON), "Detector should be an instance of SessionJSON"
        self.detector = detector
        self.bbox_padding = bbox_padding
        self.detection_cache = OrderedDict()  # Cache for storing detections by frame index
        self.max_cache_size = 100
    
    def __iter__(self):
        for window_idx, frame_indices in enumerate(self._iterate_sliding_window()):
            buffer = self.vr.get_batch(frame_indices).asnumpy()
            w, h = buffer.shape[2], buffer.shape[1]
            
            # Get the bounding boxes for the current window using caching
            annontations = self._detect_with_caching(buffer, frame_indices)
            figures = [fig for frame in annontations for fig in frame['objects']]
            for fig in figures:
                fig['geometry'] = fig.copy()
            
            # Get the maximal bounding box
            x1, y1, x2, y2 = get_maximal_bbox_crop(figures, (w, h), padding=self.bbox_padding)

            buffer = buffer[:, y1:y2, x1:x2, :]
            if window_idx == 0:
                print(f"Maximal bounding box: {x1}, {y1}, {x2}, {y2}")
                from my_utils import save_frames_as_video
                save_frames_as_video(buffer, "tmp/maximal_bbox.mp4", fps=10)
            buffer = self.data_transform(buffer)
            yield buffer, frame_indices
    
    def _detect(self, frames):
        with TemporaryDirectory() as tmpdir:
            img_paths = []
            for i, frame in enumerate(frames):
                frame_path = os.path.join(tmpdir, f"frame_{i}.jpg")
                Image.fromarray(frame).save(frame_path)
                img_paths.append(frame_path)
            anns = self.detector.inference_image_paths(img_paths)
            anns = [ann['annotation'] for ann in anns]
        return anns

    def _detect_with_caching(self, frames, frame_indices):
        """
        Detect objects in frames with caching to avoid reprocessing frames.
        This method uses a cache to store detections for frames that have already been processed.
        The cache has a confined size (e.g, 100 last frames), and older detections are removed when the cache is full.
        
        Args:
            frames: List of frame images
            frame_indices: List of frame indices in the original video
            
        Returns:
            List of annotations in Supervisely JSON format (returned by the detector)
        """
        # Determine which frames need to be processed
        frames_to_process = []
        uncached_indices = []
        positions = []
        
        for pos, frame_idx in enumerate(frame_indices):
            if frame_idx not in self.detection_cache:
                frames_to_process.append(frames[pos])
                uncached_indices.append(frame_idx)
                positions.append(pos)
        
        # If there are frames to process, detect objects
        if frames_to_process:
            new_annotations = self._detect(frames_to_process)
            
            # Update the cache with new detections
            for frame_idx, annotation in zip(uncached_indices, new_annotations):
                # If cache is at max size, remove the oldest item (first one in OrderedDict)
                if len(self.detection_cache) >= self.max_cache_size:
                    self.detection_cache.popitem(last=False)
                self.detection_cache[frame_idx] = annotation
                
        # Retrieve annotations for all requested frames
        return [self.detection_cache[idx] for idx in frame_indices]

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function that stacks frames but keeps indices as a list of lists.
        
        Args:
            batch: List of (buffer, frame_indices) tuples
            
        Returns:
            Tuple of (stacked_frames, list_of_indices)
        """
        tensors, indices = zip(*batch)
        # Stack frames into a batch tensor
        tensor_batch = torch.stack(tensors)
        # Just keep indices as a list of lists without any tensor conversion
        indices_batch = list(indices)
        
        return tensor_batch, indices_batch

def get_maximal_bbox_crop(figures, img_size, padding=0.0):
    w, h = img_size
    if not figures:
        print(f"No found any detections")
        x1, y1, x2, y2 = 0, 0, w, h
    else:
        x1, y1, x2, y2 = get_maximal_bbox(figures)
        x1, y1, x2, y2 = get_square_bbox((x1, y1, x2, y2), padding=padding)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return x1, y1, x2, y2