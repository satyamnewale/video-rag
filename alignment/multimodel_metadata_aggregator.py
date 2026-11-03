class MultimodalMetadataAggregator:

    def aggregate(self, aligned_windows):

        aggregated_records = []

        for window in aligned_windows:

            audio_events = window["audio_events"]
            ocr_events = window["ocr_events"]
            blip_events = window["blip_events"]
            yolo_events = window["yolo_events"]

            # --------------------------------
            # Audio aggregation
            # --------------------------------

            transcripts = []
            speakers = set()
            words = []

            for event in audio_events:

                text = event.get("transcript", "").strip()

                if text:
                    transcripts.append(text)

                speaker = event.get("speaker_id")

                if speaker:
                    speakers.add(speaker)

                words.extend(event.get("words", []))

            # --------------------------------
            # OCR aggregation
            # --------------------------------

            ocr_items = []
            seen_ocr = set()

            for event in ocr_events:

                for item in event.get("ocr_results", []):

                    text = item.get("text", "").strip()

                    if not text:
                        continue

                    dedupe_key = (
                        event.get("keyframe_id"),
                        text.casefold()
                    )

                    if dedupe_key in seen_ocr:
                        continue

                    seen_ocr.add(dedupe_key)

                    ocr_items.append({
                        "text": text,
                        "confidence": item.get("confidence"),
                        "bbox": item.get("bbox"),
                        "keyframe_id": event.get("keyframe_id")
                    })

            # --------------------------------
            # BLIP aggregation
            # --------------------------------

            descriptions = []

            for event in blip_events:

                description = event.get(
                    "visual_description", ""
                ).strip()

                if description:
                    descriptions.append({
                        "description": description,
                        "keyframe_id": event.get("keyframe_id"),
                        "timestamp_seconds": event.get(
                            "timestamp_seconds"
                        )
                    })

            # --------------------------------
            # YOLO aggregation
            # --------------------------------

            object_map = {}

            for event in yolo_events:

                for item in event.get(
                    "detected_objects", []
                ):

                    label = item.get("label")

                    if not label:
                        continue

                    confidence = float(
                        item.get("confidence", 0)
                    )

                    if (
                        label not in object_map
                        or confidence > object_map[label]["confidence"]
                    ):

                        object_map[label] = {
                            "label": label,
                            "class_id": item.get("class_id"),
                            "confidence": confidence,
                            "bbox": item.get("bbox"),
                            "keyframe_id": event.get("keyframe_id")
                        }

            aggregated_records.append({

                "window_id": window["window_id"],

                "scene_id": window["scene_id"],

                "start_seconds": window["start_seconds"],

                "end_seconds": window["end_seconds"],

                "transcript": " ".join(transcripts),

                "speakers": sorted(speakers),

                "words": words,

                "ocr": ocr_items,

                "visual_descriptions": descriptions,

                "detected_objects": list(object_map.values()),

                "keyframe_ids": window["keyframe_ids"]

            })

        return aggregated_records