import uuid


class StructuredVideoRecordBuilder:

    def build_records(
        self,
        aggregated_records,
        frame_references,
        video_id
    ):

        structured_records = []

        for item in aggregated_records:

            frame_refs = []

            for keyframe_id in item["keyframe_ids"]:

                reference = frame_references.get(keyframe_id)

                if reference:
                    frame_refs.append(reference)

            # Stable record ID for the same video window.
            record_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                (
                    f"{video_id}:"
                    f"{item['scene_id']}:"
                    f"{item['start_seconds']}:"
                    f"{item['end_seconds']}"
                )
            ))

            structured_records.append({

                "record_id": record_id,

                "video_id": video_id,

                "scene_id": item["scene_id"],

                "window_id": item["window_id"],

                "start_seconds": item["start_seconds"],

                "end_seconds": item["end_seconds"],

                "transcript": item["transcript"],

                "speakers": item["speakers"],

                "words": item["words"],

                "ocr": item["ocr"],

                "visual_descriptions": item[
                    "visual_descriptions"
                ],

                "detected_objects": item[
                    "detected_objects"
                ],

                "frame_references": frame_refs

            })

        return structured_records