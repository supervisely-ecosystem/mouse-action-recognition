# from decord import VideoReader, cpu
# from video_sliding_window import VideoSlidingWindow
# video_path = "/root/volume/data/mouse/HOM Mice F.2632_HOM 12 Days post tre/12 Days post tre/video/GL010560.MP4"
# vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
# frame_indices = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
# buffer = vr.get_batch(frame_indices).asnumpy()
# print(buffer.shape)

# ds = VideoSlidingWindow(video_path, num_frames=16, frame_sample_rate=2, input_size=224, stride=5)
# buffer = ds._extract_frames(frame_indices)
# print(buffer.shape)

from maximal_crop_dataset import MaximalCropDataset
from my_utils import save_frames_as_video, save_frames_as_image

ds = MaximalCropDataset(
    anno_path="data/mouse/val.csv",
    data_path="data/mouse/",
    det_anno_path="data/mouse/detections/",
    mode="train",
    clip_len=16,
    frame_sample_rate=2,
)

idx = 3000
data = ds[idx]
print(data[1:])
# save_frames_as_video(data[0], "tmp/output1.mp4", fps=10)
# save_frames_as_image(data[0], "tmp/output1.jpg")

save_frames_as_video(data[0][0], "tmp/output1.mp4", fps=12)
save_frames_as_video(data[0][1], "tmp/output2.mp4", fps=12)
save_frames_as_image(data[0][0], "tmp/output1.jpg")
save_frames_as_image(data[0][1], "tmp/output2.jpg")

vido_path = ds.dataset_samples[idx]
print(vido_path)
ann_path = ds._get_det_ann_path(vido_path)
from decord import VideoReader
vr = VideoReader(vido_path)
img = vr[0].asnumpy()
print(img.shape)
import json
with open(ann_path, "r") as f:
    ann = json.load(f)
bbox = ann["frames"][0]["figures"][0]['geometry']['points']['exterior']
x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
print(x1, y1, x2, y2)
# draw bbox
import cv2
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite("tmp/crop1.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


