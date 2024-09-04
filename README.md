**USE best.pt file to infererence INPUT_YOUTUBE_VEDIO.MP4 .Below the following code.**

import os
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('/content/runs/detect/train/weights/best.pt')

# Run video inference
results = model.predict(source='/content/downloaded_video.mp4', save=True, save_txt=True, save_conf=True)

# Create directories to save individual frames and bounding box coordinates
output_dir = 'runs/predict/exp/frames/'
os.makedirs(output_dir, exist_ok=True)

# Video capture object to retrieve frames per second (fps) information
video_capture = cv2.VideoCapture('/content/downloaded_video.mp4')
fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)  # Save 1 frame per second

# Save each frame at 1 FPS and save bounding box coordinates
for i, result in enumerate(results):
    if i % frame_interval == 0:  # Save only one frame per second
        frame = result.orig_img  # Original frame
        frame_path = f"{output_dir}/frame_{i:05d}.jpg"
        cv2.imwrite(frame_path, frame)  # Save the frame

        # Save bounding box coordinates
        bbox_dir = f"{output_dir}/bbox/"
        os.makedirs(bbox_dir, exist_ok=True)
        bbox_path = f"{bbox_dir}/frame_{i:05d}.txt"
        with open(bbox_path, 'w') as f:
            for box in result.boxes.xywh:
                x_center, y_center, width, height = box[:4]
                class_id = int(box[4]) if len(box) > 4 else 0  # Assuming class_id is at index 4
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
