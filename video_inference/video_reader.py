import cv2
import numpy as np
from typing import Optional, Tuple, Iterator, Union, Dict, List
import os

class VideoReader:
    """
    A convenient wrapper around OpenCV's VideoCapture for reading video files frame-by-frame.
    
    This class provides a more Pythonic interface to read frames, get video properties,
    and work with video files.
    """
    
    def __init__(self, video_path: str):
        """
        Initialize the VideoReader with a video file path.
        
        Args:
            video_path (str): Path to the video file to read
        
        Raises:
            FileNotFoundError: If the video file doesn't exist
            ValueError: If the video file couldn't be opened
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Cache video properties
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_frame_pos = 0
        
    def __enter__(self):
        """Context manager entry point"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point"""
        self.release()
    
    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterate through all frames in the video.
        
        Returns:
            Iterator[np.ndarray]: Iterator yielding frames as numpy arrays
        """
        self.reset()
        return self
    
    def __next__(self) -> np.ndarray:
        """Get the next frame when iterating"""
        ret, frame = self.cap.read()
        if not ret:
            self.reset()  # Reset for future iterations
            raise StopIteration
        self._current_frame_pos += 1
        return frame
    
    def __len__(self) -> int:
        """Return the total number of frames"""
        return self._frame_count
    
    @property
    def width(self) -> int:
        """Get the video width in pixels"""
        return self._width
    
    @property
    def height(self) -> int:
        """Get the video height in pixels"""
        return self._height
    
    @property
    def fps(self) -> float:
        """Get the video framerate"""
        return self._fps
    
    @property
    def frame_count(self) -> int:
        """Get the total number of frames in the video"""
        return self._frame_count
    
    @property
    def duration(self) -> float:
        """Get the video duration in seconds"""
        return self._frame_count / self._fps if self._fps else 0
    
    @property
    def position(self) -> int:
        """Get current frame position"""
        return self._current_frame_pos
    
    @property
    def metadata(self) -> Dict:
        """Return a dictionary with all video metadata"""
        return {
            'width': self._width,
            'height': self._height,
            'fps': self._fps,
            'frame_count': self._frame_count,
            'duration': self.duration,
            'path': self.video_path,
            'fourcc': self.get_fourcc()
        }
    
    def get_fourcc(self) -> str:
        """Get the four character code (codec) as a string"""
        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        return "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the video.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: A tuple containing a success flag and the frame
                                             (None if reading failed)
        """
        ret, frame = self.cap.read()
        if ret:
            self._current_frame_pos += 1
        return ret, frame
    
    def read_frame(self) -> np.ndarray:
        """
        Read the next frame, raising StopIteration if at the end.
        
        Returns:
            np.ndarray: The next frame
            
        Raises:
            StopIteration: When no more frames are available
        """
        ret, frame = self.read()
        if not ret:
            raise StopIteration("No more frames")
        return frame
    
    def seek(self, frame_number: int) -> bool:
        """
        Seek to a specific frame in the video.
        
        Args:
            frame_number (int): Frame number to seek to (0-indexed)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if 0 <= frame_number < self._frame_count:
            ret = self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            if ret:
                self._current_frame_pos = frame_number
            return ret
        return False
    
    def seek_time(self, seconds: float) -> bool:
        """
        Seek to a specific time in the video.
        
        Args:
            seconds (float): Time in seconds to seek to
            
        Returns:
            bool: True if successful, False otherwise
        """
        frame_number = int(seconds * self._fps)
        return self.seek(frame_number)
    
    def reset(self) -> bool:
        """
        Reset to the beginning of the video.
        
        Returns:
            bool: True if successful, False otherwise
        """
        return self.seek(0)
    
    def release(self) -> None:
        """Release the video capture resources"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def read_frames_batch(self, batch_size: int) -> List[np.ndarray]:
        """
        Read a batch of frames at once.
        
        Args:
            batch_size (int): Number of frames to read
            
        Returns:
            List[np.ndarray]: List of frames (may be shorter than batch_size if end of video is reached)
        """
        frames = []
        for _ in range(batch_size):
            try:
                frames.append(self.read_frame())
            except StopIteration:
                break
        return frames
    
    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by its frame number.
        
        Args:
            frame_number (int): Frame number to get (0-indexed)
            
        Returns:
            Optional[np.ndarray]: The requested frame or None if not available
        """
        current_pos = self._current_frame_pos
        if self.seek(frame_number):
            ret, frame = self.read()
            # Restore previous position
            self.seek(current_pos)
            return frame if ret else None
        return None
    
    def get_frame_at_time(self, seconds: float) -> Optional[np.ndarray]:
        """
        Get a specific frame by timestamp.
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            Optional[np.ndarray]: The requested frame or None if not available
        """
        frame_number = int(seconds * self._fps)
        return self.get_frame_at(frame_number)
