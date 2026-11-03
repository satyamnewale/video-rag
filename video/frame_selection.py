
import cv2
import numpy as np
from collections import defaultdict


class FrameSelectionEngine:

    def __init__(
        self,
        max_frames_per_scene=5,
        temporal_bins=5
    ):
        self.max_frames_per_scene = max_frames_per_scene
        self.temporal_bins = temporal_bins

    # Calculate frame sharpness
    def calculate_sharpness(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )

        return cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

    # Select representative frames
    def select_keyframes(self, candidate_frames, scenes):

        scene_candidates = defaultdict(list)

        for candidate in candidate_frames:
            scene_candidates[candidate["scene_id"]].append(
                candidate
            )

        selected_keyframes = []

        for scene in scenes:

            scene_id = scene["scene_id"]
            start = scene["start_seconds"]
            end = scene["end_seconds"]

            candidates = scene_candidates[scene_id]

            if not candidates:
                continue

            # Divide scene into temporal bins
            bin_count = min(
                self.temporal_bins,
                self.max_frames_per_scene
            )

            bin_width = (end - start) / bin_count

            for bin_index in range(bin_count):

                bin_start = start + bin_index * bin_width
                bin_end = bin_start + bin_width

                bin_candidates = [
                    frame for frame in candidates
                    if bin_start <= frame["timestamp_seconds"] < bin_end
                ]

                if not bin_candidates:
                    continue

                # Calculate sharpness
                for frame in bin_candidates:
                    frame["sharpness_score"] = (
                        self.calculate_sharpness(frame["frame"])
                    )

                # Choose sharpest frame
                best_frame = max(
                    bin_candidates,
                    key=lambda x: x["sharpness_score"]
                )

                selected_keyframes.append(best_frame)

        return selected_keyframes
