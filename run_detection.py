import cv2
import numpy as np


def extract_frames_from_video(video_path: str, num_frames_to_sample: int = 15):
    """
    Extracts frames from the video at specified intervals.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video -> {video_path}")
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Error: No frames found in video -> {video_path}")
        cap.release()
        return frames

    if total_frames < num_frames_to_sample:
        frame_indices = np.arange(total_frames)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    print(f"{len(frames)} frames successfully extracted from video.")
    return frames 