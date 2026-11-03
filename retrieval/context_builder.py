import json


class ContextBuilder:

    # -----------------------------------------
    # Decode ChromaDB JSON metadata
    # -----------------------------------------

    @staticmethod
    def _decode_json(value):

        if isinstance(value, str):

            try:
                return json.loads(value)

            except json.JSONDecodeError:
                return value

        return value

    # -----------------------------------------
    # Format timestamp
    # -----------------------------------------

    @staticmethod
    def _format_timestamp(seconds):

        seconds = float(seconds)

        hours = int(seconds // 3600)

        minutes = int((seconds % 3600) // 60)

        remaining_seconds = seconds % 60

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{remaining_seconds:05.2f}"
        )

    # -----------------------------------------
    # Build one record's context
    # -----------------------------------------

    def _format_record(self, document, index):

        metadata = document.get("metadata", {})

        start_seconds = float(
            metadata.get("start_seconds", 0)
        )

        end_seconds = float(
            metadata.get("end_seconds", 0)
        )

        frame_references = self._decode_json(
            metadata.get("frame_references", [])
        )

        speakers = self._decode_json(
            metadata.get("speakers", [])
        )

        ocr = self._decode_json(
            metadata.get("ocr", [])
        )

        visual_descriptions = self._decode_json(
            metadata.get("visual_descriptions", [])
        )

        detected_objects = self._decode_json(
            metadata.get("detected_objects", [])
        )

        start_time = self._format_timestamp(
            start_seconds
        )

        end_time = self._format_timestamp(
            end_seconds
        )

        record_text = f"""
SOURCE {index}

Video ID: {metadata.get("video_id", "unknown")}

Scene ID: {metadata.get("scene_id", "unknown")}

Time: {start_time} - {end_time}

Transcript and indexed text:
{document.get("text", "")}

Speakers:
{json.dumps(speakers, ensure_ascii=False)}

OCR:
{json.dumps(ocr, ensure_ascii=False)}

Visual descriptions:
{json.dumps(visual_descriptions, ensure_ascii=False)}

Detected objects:
{json.dumps(detected_objects, ensure_ascii=False)}

Frame references:
{json.dumps(frame_references, ensure_ascii=False)}

Reranking score:
{document.get("rerank_score", "N/A")}
"""

        source = {

            "source_number": index,

            "record_id": document.get("record_id"),

            "video_id": metadata.get("video_id"),

            "scene_id": metadata.get("scene_id"),

            "start_seconds": start_seconds,

            "end_seconds": end_seconds,

            "start_timestamp": start_time,

            "end_timestamp": end_time,

            "frame_references": frame_references,

            "rerank_score": document.get("rerank_score")

        }

        return record_text.strip(), source

    # -----------------------------------------
    # Build final context
    # -----------------------------------------

    def build_context(self, documents):

        if not documents:
            return "", []

        context_parts = []

        sources = []

        for index, document in enumerate(
            documents,
            start=1
        ):

            record_text, source = self._format_record(
                document,
                index
            )

            context_parts.append(record_text)

            sources.append(source)

        context = "\n\n".join(context_parts)

        return context, sources