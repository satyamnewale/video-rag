import cv2
import numpy as np


class PerceptualHashProcessor:

    def __init__(self, hash_size=8, threshold=8):
        self.hash_size = hash_size
        self.threshold = threshold

    def generate_hash(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        resized = cv2.resize(gray, (32, 32))

        dct = cv2.dct(np.float32(resized))

        low_freq = dct[:self.hash_size, :self.hash_size]

        # Exclude DC coefficient
        values = low_freq.flatten()[1:]

        median = np.median(values)
        bits = values > median

        hash_value = 0

        for bit in bits:
            hash_value = (hash_value << 1) | int(bit)

        return hash_value

    def hamming_distance(self, hash1, hash2):
        return (hash1 ^ hash2).bit_count()

    def remove_duplicates(self, keyframes, video_id="video"):

        unique_frames = []
        seen_hashes = {}

        for frame in keyframes:

            scene_id = frame["scene_id"]

            frame_number = frame["frame_number"]

            frame_hash = self.generate_hash(frame["frame"])

            frame["perceptual_hash"] = format(
                frame_hash, "016x"
            )

            scene_hashes = seen_hashes.get(scene_id, [])

            is_duplicate = any(
                self.hamming_distance(frame_hash, previous_hash)
                <= self.threshold
                for previous_hash in scene_hashes
            )

            if is_duplicate:
                continue

            # Register this frame as unique.
            seen_hashes.setdefault(scene_id, []).append(frame_hash)

            # Stable ID based on video, scene and original frame number.
            frame["keyframe_id"] = (
                f"{video_id}_scene_{scene_id:04d}"
                f"_frame_{frame_number:08d}"
            )

            unique_frames.append(frame)

        return unique_frames