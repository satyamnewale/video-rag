class TextNormalizationProcessor:

    def normalize_record(self, record):

        sections = []

        # --------------------------------
        # Video information
        # --------------------------------

        sections.append(
            f"Video ID: {record['video_id']}"
        )

        sections.append(
            f"Scene ID: {record['scene_id']}"
        )

        sections.append(
            (
                f"Time: {record['start_seconds']:.2f} "
                f"to {record['end_seconds']:.2f} seconds"
            )
        )

        # --------------------------------
        # Transcript
        # --------------------------------

        transcript = record.get("transcript", "").strip()

        if transcript:

            sections.append(
                f"Audio Transcript: {transcript}"
            )

        # --------------------------------
        # Speakers
        # --------------------------------

        speakers = record.get("speakers", [])

        if speakers:

            sections.append(
                f"Speakers: {', '.join(speakers)}"
            )

        # --------------------------------
        # OCR text
        # --------------------------------

        ocr_texts = [
            item["text"]
            for item in record.get("ocr", [])
            if item.get("text")
        ]

        if ocr_texts:

            sections.append(
                f"On-Screen Text: {'; '.join(ocr_texts)}"
            )

        # --------------------------------
        # Visual descriptions
        # --------------------------------

        descriptions = [
            item["description"]
            for item in record.get("visual_descriptions", [])
            if item.get("description")
        ]

        if descriptions:

            sections.append(
                f"Visual Descriptions: {'; '.join(descriptions)}"
            )

        # --------------------------------
        # Detected objects
        # --------------------------------

        objects = [
            item["label"]
            for item in record.get("detected_objects", [])
            if item.get("label")
        ]

        if objects:

            sections.append(
                f"Detected Objects: {', '.join(sorted(set(objects)))}"
            )

        return "\n".join(sections)

    def normalize_records(self, records):

        normalized_records = []

        for record in records:

            text = self.normalize_record(record)

            if not text.strip():
                continue

            normalized_records.append({

                "record_id": record["record_id"],

                "video_id": record["video_id"],

                "scene_id": record["scene_id"],

                "start_seconds": record["start_seconds"],

                "end_seconds": record["end_seconds"],

                "text": text,

                "metadata_source": record

            })

        return normalized_records