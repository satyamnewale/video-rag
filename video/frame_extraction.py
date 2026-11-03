
import cv2
import numpy as np


class FrameExtractionProcessor:

    def __init__(
        self,
        sample_interval=1.0,
        target_size=(640, 640),
        min_blur_score=10.0
    ):
        self.sample_interval = sample_interval
        self.target_size = target_size
        self.min_blur_score = min_blur_score

    # --------------------------------
    # 1. Extract frames from a scene
    # --------------------------------
    def extract_frames(self, video_path, scenes):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Unable to open video")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            cap.release()
            raise ValueError("Invalid video FPS")

        candidate_frames = []

        try:
            for scene in scenes:

                scene_id = scene["scene_id"]

                start_frame = scene["start_frame"]
                end_frame = scene["end_frame"]

                # Convert sampling interval into frame interval
                frame_interval = max(
                    1, int(round(fps * self.sample_interval))
                )

                for frame_number in range(
                    start_frame, end_frame, frame_interval
                ):

                    # Move video reader to target frame
                    cap.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        frame_number
                    )

                    success, frame = cap.read()

                    if not success:
                        continue

                    # Calculate timestamp
                    timestamp = frame_number / fps

                    # Preprocess frame
                    processed_frame = self.preprocess_frame(frame)

                    if processed_frame is None:
                        continue

                    candidate_frames.append({
                        "scene_id": scene_id,
                        "frame_number": frame_number,
                        "timestamp_seconds": timestamp,
                        "timestamp": self.format_timestamp(timestamp),
                        "frame": processed_frame
                    })

        finally:
            cap.release()

        return candidate_frames

    # --------------------------------
    # 2. Preprocess extracted frames
    # --------------------------------
    def preprocess_frame(self, frame):

        if frame is None or frame.size == 0:
            return None

        # Resize while maintaining aspect ratio
        height, width = frame.shape[:2]

        scale = min(
            self.target_size[0] / width,
            self.target_size[1] / height
        )

        new_width = int(width * scale)
        new_height = int(height * scale)

        resized = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        # Check image sharpness
        gray = cv2.cvtColor(
            rgb_frame,
            cv2.COLOR_RGB2GRAY
        )

        blur_score = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        if blur_score < self.min_blur_score:
            return None

        return rgb_frame

    # --------------------------------
    # 3. Format timestamp
    # --------------------------------
    def format_timestamp(self, seconds):

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60

        return f"{hours:02}:{minutes:02}:{secs:06.3f}"
