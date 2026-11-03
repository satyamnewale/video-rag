from pathlib import Path
import cv2


class FrameStorageProcessor:

    def __init__(self, storage_dir="storage_data/frames"):

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def save_frames(self, unique_frames, video_id):

        video_dir = self.storage_dir / video_id

        video_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        frame_references = {}

        for frame_data in unique_frames:

            keyframe_id = frame_data["keyframe_id"]

            frame = frame_data["frame"]

            output_path = video_dir / f"{keyframe_id}.jpg"

            # Frames are RGB; OpenCV imwrite expects BGR.
            bgr_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )

            success = cv2.imwrite(
                str(output_path),
                bgr_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            if not success:
                raise RuntimeError(
                    f"Failed to save frame: {keyframe_id}"
                )

            frame_references[keyframe_id] = {
                "keyframe_id": keyframe_id,
                "frame_path": str(output_path.resolve()),
                "scene_id": frame_data["scene_id"],
                "frame_number": frame_data["frame_number"],
                "timestamp_seconds": frame_data["timestamp_seconds"],
                "timestamp": frame_data["timestamp"]
            }

        return frame_references