from decord import VideoReader, cpu
from video_sliding_window import VideoSlidingWindow
video_path = "/root/volume/data/mouse/HOM Mice F.2632_HOM 12 Days post tre/12 Days post tre/video/GL010560.MP4"
vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
frame_indices = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
buffer = vr.get_batch(frame_indices).asnumpy()
print(buffer.shape)

ds = VideoSlidingWindow(video_path, num_frames=16, frame_sample_rate=2, input_size=224, stride=5)
buffer = ds._extract_frames(frame_indices)
print(buffer.shape)