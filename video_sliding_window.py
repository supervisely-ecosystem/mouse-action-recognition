from decord import VideoReader, cpu
import torch
from torch.utils.data import Dataset, IterableDataset
import video_transforms as video_transforms
import volume_transforms as volume_transforms


class VideoSlidingWindow(IterableDataset):

    def __init__(self, video_path, num_frames=16,
                 frame_sample_rate=2, input_size=224,
                 stride=5):
        self.video_path = video_path
        self.num_frames = num_frames
        self.frame_sample_rate = frame_sample_rate  # step of frame sampling
        self.input_size = input_size
        self.stride = stride
        
        self.vr = VideoReader(video_path)
            
        self.data_transform = video_transforms.Compose([
            video_transforms.Resize(size=(self.input_size, self.input_size), interpolation='bilinear'),
            # video_transforms.CenterCrop(size=(self.input_size, self.input_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        ])

    def __iter__(self):
        total_frames = len(self.vr)
        # Calculate effective frame indices considering sample rate
        max_frame_idx = total_frames - 1
        
        # Determine how many complete windows we can extract
        window_length = self.num_frames * self.frame_sample_rate
        num_windows = max(0, (total_frames - window_length) // (self.stride * self.frame_sample_rate) + 1)
        
        for window_idx in range(num_windows):
            start_idx = window_idx * self.stride * self.frame_sample_rate
            
            # Get the actual frame indices considering the sample rate
            frame_indices = [start_idx + (i * self.frame_sample_rate) for i in range(self.num_frames)]
            
            # Double-check that no frame indices are out of bounds
            if any(idx > max_frame_idx for idx in frame_indices):
                break
                
            # Extract frames and process the window
            buffer = self._extract_frames(frame_indices)
            yield buffer, frame_indices

    def _extract_frames(self, frame_indices):
        # Extract frames at the specified indices
        buffer = self.vr.get_batch(frame_indices).asnumpy()
        buffer = self.data_transform(buffer)
        return buffer

    def __len__(self):
        total_frames = len(self.vr)
        window_length = self.num_frames * self.frame_sample_rate
        num_windows = max(0, (total_frames - window_length) // (self.stride * self.frame_sample_rate) + 1)
        return num_windows

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function that stacks frames but keeps indices as a list of lists.
        
        Args:
            batch: List of (buffer, frame_indices) tuples
            
        Returns:
            Tuple of (stacked_frames, list_of_indices)
        """
        frames, indices = zip(*batch)
        # Stack frames into a batch tensor
        frames_batch = torch.stack(frames)
        # Just keep indices as a list of lists without any tensor conversion
        indices_batch = list(indices)
        
        return frames_batch, indices_batch