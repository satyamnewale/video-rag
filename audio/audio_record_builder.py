
import uuid

class StructuredAudioRecordBuilder:

    def build_records(
        self,
        aligned_result,
        video_id
    ):

        structured_records = []

        language = aligned_result.get("language")

        for index, segment in enumerate(
            aligned_result.get("segments", [])
        ):

            start = segment.get("start")
            end = segment.get("end")

            if start is None or end is None:
                continue

            text = segment.get("text", "").strip()

            if not text:
                continue

            speaker = segment.get("speaker")

            words = []

            for word_data in segment.get("words", []):

                word_start = word_data.get("start")
                word_end = word_data.get("end")

                # Some words cannot be aligned
                if word_start is None or word_end is None:
                    continue

                words.append({

                    "word": word_data.get(
                        "word", ""
                    ).strip(),

                    "start_seconds": word_start,

                    "end_seconds": word_end,

                    "confidence": word_data.get(
                        "score"
                    ),

                    "speaker_id": word_data.get(
                        "speaker", speaker
                    )

                })

            record_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{video_id}:audio:{index}:{start}:{end}"
            ))

            structured_records.append({

                "record_id": record_id,

                "video_id": video_id,

                "segment_id": index + 1,

                "start_seconds": start,

                "end_seconds": end,

                "transcript": text,

                "language": language,

                "speaker_id": speaker,

                "words": words

            })

        return structured_records
