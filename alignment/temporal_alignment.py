from collections import defaultdict


class TemporalAlignmentEngine:

    def __init__(self, window_size=2.0):

        if window_size <= 0:
            raise ValueError("window_size must be greater than zero")

        self.window_size = window_size

    # -----------------------------------------
    # Get audio segments overlapping a window
    # -----------------------------------------
    def _get_audio_events(
        self,
        audio_records,
        window_start,
        window_end
    ):

        matched = []

        for record in audio_records:

            start = record["start_seconds"]
            end = record["end_seconds"]

            # Check temporal overlap
            if start < window_end and end > window_start:

                matched.append(record)

        return matched

    # -----------------------------------------
    # Get visual events inside a window
    # -----------------------------------------
    def _get_visual_events(
        self,
        records,
        window_start,
        window_end
    ):

        matched = []

        for record in records:

            timestamp = record.get("timestamp_seconds")

            if timestamp is None:
                continue

            if window_start <= timestamp < window_end:
                matched.append(record)

        return matched

    # -----------------------------------------
    # Build windows for each scene
    # -----------------------------------------
    def align(
        self,
        scenes,
        audio_records,
        ocr_results,
        blip_results,
        yolo_results
    ):

        aligned_windows = []

        for scene in scenes:

            scene_id = scene["scene_id"]

            scene_start = float(scene["start_seconds"])
            scene_end = float(scene["end_seconds"])

            window_start = scene_start
            window_index = 1

            while window_start < scene_end:

                window_end = min(
                    window_start + self.window_size,
                    scene_end
                )

                # Audio uses interval overlap.
                audio_events = self._get_audio_events(
                    audio_records,
                    window_start,
                    window_end
                )

                # Visual records use frame timestamps.
                scene_ocr = [
                    item for item in ocr_results
                    if item["scene_id"] == scene_id
                ]

                scene_blip = [
                    item for item in blip_results
                    if item["scene_id"] == scene_id
                ]

                scene_yolo = [
                    item for item in yolo_results
                    if item["scene_id"] == scene_id
                ]

                matched_ocr = self._get_visual_events(
                    scene_ocr,
                    window_start,
                    window_end
                )

                matched_blip = self._get_visual_events(
                    scene_blip,
                    window_start,
                    window_end
                )

                matched_yolo = self._get_visual_events(
                    scene_yolo,
                    window_start,
                    window_end
                )

                # Collect unique keyframe IDs.
                keyframe_ids = set()

                for collection in (
                    matched_ocr,
                    matched_blip,
                    matched_yolo
                ):

                    for item in collection:

                        keyframe_id = item.get("keyframe_id")

                        if keyframe_id:
                            keyframe_ids.add(keyframe_id)

                aligned_windows.append({

                    "scene_id": scene_id,

                    "window_id": (
                        f"scene_{scene_id:04d}_window_{window_index:04d}"
                    ),

                    "start_seconds": window_start,

                    "end_seconds": window_end,

                    "audio_events": audio_events,

                    "ocr_events": matched_ocr,

                    "blip_events": matched_blip,

                    "yolo_events": matched_yolo,

                    "keyframe_ids": sorted(keyframe_ids)

                })

                window_start = window_end
                window_index += 1

        return aligned_windows